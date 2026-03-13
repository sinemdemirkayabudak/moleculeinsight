# MoleculeInsight Architecture

> **For Users:** Start with [README.md](../README.md) for installation and feature overview.  
> **For Developers:** This document goes deep into module design, algorithms, and testing strategy.

## When to Read This Document

- You're contributing new features or fixing bugs
- You want to understand how modules interact
- You need detailed algorithm explanations
- You're reviewing test coverage strategy

## Project Structure

```
moleculeinsight/
├── app/                          # Main application code
│   ├── __init__.py
│   ├── config.py                 # Configuration, logging, environment setup
│   ├── validators.py             # SMILES input validation
│   ├── molecule.py               # RDKit molecular operations & Lipinski rules
│   ├── pubchem.py                # PubChem API integration (cached for 24h)
│   ├── chembl.py                 # ChEMBL API integration & bioactivity lookup (cached for 24h)
│   ├── similarity_search/        # Molecular similarity module
│   │   ├── __init__.py
│   │   ├── cli.py                # Command-line interface & pipeline orchestration
│   │   ├── pipeline.py           # Search pipeline: fingerprints → similarities
│   │   ├── fingerprints.py       # Morgan fingerprint computation
│   │   ├── validators.py         # Input validation for search
│   │   └── visualization.py      # Structure images & ranking plots (cached)
│   ├── qsar/                     # Quantitative Structure-Activity Relationships (QSAR)
│   │   ├── __init__.py
│   │   ├── train_models.py       # Train RF + XGBoost models on ChEMBL EGFR data (100% coverage)
│   │   ├── model_visualizations.py # Generate performance plots for dashboard (92% coverage)
│   │   ├── explain.py            # SHAP explainability engine for feature importance (100% coverage)
│   │   ├── visualize.py          # SHAP visualization utilities (100% coverage)
│   │   ├── qsar_prediction.py    # QSAR pipeline for data loading & preprocessing (100% coverage)
│   │   ├── predict.py            # Model prediction interface (100% coverage)
│   │   ├── features.py           # Feature computation (Morgan + RDKit) (100% coverage)
│   │   ├── preprocessing.py      # Data cleaning & splitting (100% coverage)
│   │   ├── data_loader.py        # Load ChEMBL EGFR IC50 data (100% coverage)
│   │   ├── saved_models/         # Trained model artifacts
│   │   │   ├── egfr_rf_model.pkl
│   │   │   ├── egfr_xgb_model.pkl
│   │   │   ├── egfr_metadata.json
│   │   │   ├── egfr_performance.json
│   │   │   └── egfr_feature_annotations.json
│   │   └── visualizations/       # Performance plot outputs
│   │       ├── 01_residuals.png
│   │       ├── 02_predictions_vs_actual.png
│   │       ├── 03_feature_importance.png
│   │       ├── 04_error_distribution.png
│   │       ├── 05_model_summary.png
│   │       └── 06_shap_heatmap.png
│   ├── utils.py                  # Utility functions (safe execution, API calls)
│   ├── ui.py                     # Streamlit UI components (refactored into helpers)
│   ├── data/                     # Sample data for demonstrations
│   │   └── sample/
│   │       ├── DATA.md                             # Sample data documentation
│   │       ├── query_molecules.csv                 # Similarity search queries (5 molecules)
│   │       ├── reference_library.csv               # Similarity search reference library (9 molecules)
│   │       ├── moleculeinsight_test_dataset.csv    # Edge case test data (26 molecules)
│   │       └── lipinski_test_dataset.csv           # Lipinski rule test data (16 molecules)
│   └── components/
│       └── cards.py              # Reusable UI card components
├── tests/                        # Comprehensive test suite (452 tests, 76.05% coverage)
│   ├── conftest.py               # Pytest configuration & fixtures
│   ├── test_validators.py        # Input validation tests (14 tests)
│   ├── test_molecule.py          # Molecular analysis tests (16 tests)
│   ├── test_pubchem.py           # PubChem integration tests (13 tests)
│   ├── test_chembl.py            # ChEMBL integration tests (17 tests)
│   ├── test_utils.py             # Utility function tests (8 tests)
│   ├── test_similarity_search_fingerprints.py      # Fingerprint tests (11 tests)
│   ├── test_similarity_search_pipeline.py          # Pipeline integration tests (49 tests)
│   ├── test_similarity_search_validators.py        # Validator tests (12 tests)
│   ├── test_similarity_search_visualization.py     # Visualization tests (22 tests)
│   ├── test_similarity_search_cli_basic.py         # Basic CLI tests (40 tests)
│   ├── test_similarity_search_cli_visualization.py # CLI visualization tests (11 tests)
│   ├── test_similarity_search_cli_edge_cases.py    # CLI edge cases & exceptions (44 tests)
│   ├── test_qsar_data_loader.py                   # QSAR data loading (7 tests)
│   ├── test_qsar_feature_computation.py           # Feature computation (8 tests)
│   ├── test_qsar_feature_prepare.py               # Feature preprocessing (11 tests)
│   ├── test_qsar_preprocessing.py                 # Data preprocessing (7 tests)
│   ├── test_qsar_train_models.py                  # Model training (19 tests, 100% coverage)
│   ├── test_qsar_model_visualizations.py          # Model visualizations (20 tests, 85.7% coverage)
│   ├── test_qsar_predict.py                       # Model predictions (8 tests)
│   ├── test_qsar_prediction.py                    # Pipeline predictions (15 tests)
│   ├── test_qsar_prediction_advanced.py           # Advanced predictions (8 tests)
│   ├── test_qsar_explain.py                       # SHAP explanations (6 tests)
│   ├── test_qsar_visualize.py                     # SHAP visualizations (7 tests)
│   └── components/
│       └── __init__.py           # Component test utilities
├── .github/
│   └── workflows/
│       └── tests.yml             # GitHub Actions CI/CD pipeline
├── main.py                       # Application entry point
├── pyproject.toml                # Project configuration & dependencies
└── README.md                     # User documentation
```

## Module Descriptions

### Core Modules (100% Test Coverage ✅)

**`config.py`**
- Environment variable loading
- Logging configuration
- API endpoint setup

**`validators.py`**
- SMILES string validation
- Input sanitization
- Length and character checks

**`molecule.py`**
- RDKit molecular operations
- Property calculations (MW, LogP, TPSA, HBD, HBA, rotatable bonds)
- Lipinski's Rule-of-5 evaluation

**`pubchem.py`**
- PubChem API queries via PubChemPy
- Molecular metadata retrieval (IUPAC name, synonyms, CID)
- Common name cleaning and normalization

**`chembl.py`**
- ChEMBL API integration
- Molecular lookup by InChIKey
- Bioactivity data retrieval and filtering
- Activity type, units, and confidence scoring

**`utils.py`**
- Safe function execution with error handling
- HTTP requests with timeout/retry logic
- JSON response parsing

### Similarity Search Module (100% Test Coverage - All 5 Submodules ✅)

**`cli.py`** (100% coverage ✅ - 10 tests)
- Command-line interface and argument parsing
- Main entry point with comprehensive error handling
- CSV file loading and DataFrame input support
- Pipeline orchestration
- Exception handling for ValueError, FileNotFoundError, and generic exceptions

**`fingerprints.py`** (100% coverage ✅ - 11 tests)
- Morgan fingerprint computation (ECFP4 with configurable radius 0-5)
- Tanimoto similarity metric for molecular comparison

**`pipeline.py`** (100% coverage ✅ - 27+ tests)
- SMILES to molecules conversion with filtering
- Fingerprint computation and caching
- Similarity search with pairwise comparisons
- Result ranking and top-N selection
- Exception handling for empty queries and reference libraries
- Debug logging paths for multi-query processing

**`validators.py`** (100% coverage ✅ - 9 tests)
- Parameter validation (radius, top_n ranges)
- DataFrame validation (required columns, data integrity)

**`visualization.py`** (96.51% coverage - 20+ tests)
- Molecular structure image generation (side-by-side comparison with labels)
- LRU caching for performance (up to 1024 cached images)
- Distribution plot generation for ranking visualization
- CSV export with proper column ordering
- Exception handling for invalid SMILES strings and image creation errors
- *Note: 3 lines uncovered (187-189) are logging statements in exception handler*

### QSAR Module (100% Test Coverage ✅ - All 10 Core Submodules)

**Quantitative Structure-Activity Relationships (QSAR)** - Predictive bioactivity modeling
- **Purpose:** Train and deploy machine learning models to predict molecular bioactivity (pIC50 for EGFR kinase inhibitors)
- **Data Source:** Real ChEMBL EGFR IC50 bioactivity measurements
- **Models:** Random Forest and XGBoost with rigorous cross-validation (5-fold CV, train/test splitting)
- **Performance:** Best model (XGBoost) R² = 0.7018 on test set
- **Features:** 2056-bit Morgan fingerprints + 10 RDKit descriptors for interpretability

**`data_loader.py`** (100% coverage ✅ - 7 tests)
- Load ChEMBL EGFR IC50 bioactivity data
- Filter by confidence, quality, and activity type
- Handle missing values and outliers
- Return preprocessed DataFrames with SMILES and pIC50

**`features.py`** (100% coverage ✅ - 8 tests)
- Compute Morgan fingerprints (ECFP4, radius 2, 2048 bits) with RDKit
- Compute RDKit descriptors (MW, LogP, TPSA, HBD, HBA, rotatable bonds, heteroatoms, rings)
- Stack features horizontally for model input
- Handle RDKit validation and error cases

**`preprocessing.py`** (100% coverage ✅ - 11 tests)
- Train/test split with random state for reproducibility
- Standard scaling of RDKit descriptors (preserve Morgan fingerprints as binary)
- Handle data imbalance and outlier detection
- Return preprocessed feature matrices and target vectors

**`qsar_prediction.py`** (100% coverage ✅ - 15 tests)
- Orchestrate data loading, preprocessing, and feature computation
- Encapsulate the full pipeline for reusability
- Support both batch predictions and single molecule analysis
- Return structured results (predictions, confidence intervals, feature importance)

**`train_models.py`** (100% coverage ✅ - 15 tests)
- Train Random Forest model with cross-validation evaluation
- Train XGBoost model with hyperparameter tuning
- Compute test set R², RMSE, MAE for model comparison
- Serialize trained models to `qsar/saved_models/`
- Save model metadata, performance metrics, and feature lists

**`explain.py`** (100% coverage ✅ - 6 tests)
- Generate SHAP TreeExplainers for feature importance
- Compute SHAP values for individual predictions
- Support both local (sample-level) and global (feature-level) explanations
- Cache explainers to avoid recomputation

**`visualize.py`** (100% coverage ✅ - 7 tests)
- Create SHAP force plots (individual prediction explanations)
- Create SHAP summary plots (global feature importance via mean |SHAP|)
- Create dependence plots (feature interaction visualization)
- Generate matplotlib figures optimized for Streamlit rendering
- Cache plots for performance

**`predict.py`** (100% coverage ✅ - 8 tests)
- Load trained models from disk
- Make predictions on new molecules
- Generate predictions + confidence intervals (via ensemble, residual distribution)
- Return prediction results with model metadata

**`model_visualizations.py`** (85.71% coverage ✅ - 20 tests)
- Generate comprehensive performance plots for Streamlit dashboard:
  - Residuals plot (predictions - actual values)
  - Calibration plot (predicted vs actual pIC50)
  - Feature importance bar chart (top 20 RDKit descriptors)
  - Error distribution histogram
  - Model performance summary statistics table
  - SHAP heatmap (top features across prediction samples)
- Annotate Morgan fingerprint bits with substructure information
- Save all plots to `qsar/visualizations/` for dashboard display

**Test Files** (102 QSAR-related tests across 11 modules)
- `test_qsar_data_loader.py` - Data loading and filtering (7 tests)
- `test_qsar_feature_computation.py` - Feature generation (8 tests)
- `test_qsar_feature_prepare.py` - Data preprocessing (11 tests)
- `test_qsar_preprocessing.py` - Pipeline integration (7 tests)
- `test_qsar_train_models.py` - Model training & evaluation (19 tests, 100% coverage)
- `test_qsar_model_visualizations.py` - Dashboard visualizations (20 tests)
- `test_qsar_predict.py` - Model predictions (8 tests)
- `test_qsar_prediction.py` - QSAR pipeline (15 tests)
- `test_qsar_prediction_advanced.py` - Advanced scenarios (8 tests)
- `test_qsar_explain.py` - Feature importance (6 tests)
- `test_qsar_visualize.py` - Plot generation (7 tests)

### UI Layer

**`ui.py`** - Streamlit Application (Refactored for Separation of Concerns)
- `render_single_molecule()` - Single molecule analysis interface
  - Properties tab: molecular visualization and calculations
  - Lipinski rules tab: drug-likeness compliance check
  - Bioactivity tab: ChEMBL bioactivity data (safe access with `.get()`)
  - **EGFR pIC50 Prediction** - Predict bioactivity using trained QSAR model with confidence visualization
- `render_similarity_search()` - Similarity search interface
  - File upload: CSV input handling (in-memory) or sample data loading
  - Parameter controls: fingerprint radius, top N results, ranking plots toggle
  - Results display: similarity scores table with structure images
- Refactored helper functions for better maintainability:
  - `generate_structure_images()` - Generate cached structure images for results
  - `prepare_display_dataframe()` - Transform and format DataFrame for UI display
  - `process_similarity_results()` - Orchestrate image generation and DataFrame preparation
- `display_results_table()` - Safe column dropping with `errors='ignore'`
- `display_ranking_plots()` - Query selector and matplotlib figure rendering
- `render_app()` - Main app layout and page routing

**Improvements:**
- **Type Hints** - Full type annotations on all functions
- **Defensive Programming** - Safe dict access, graceful error handling
- **Function Decomposition** - Each function has a single responsibility
- **DataFrame Safety** - Input copy prevents mutation, safe column operations

**`components/cards.py`**
- Reusable metric card components
- Styled HTML/CSS rendering

## Data Flow Architecture

### Step 1: Molecule Input & Visualization
```
User Input (SMILES)
    ↓
validators.py (SMILES validation)
    ↓
molecule.py (RDKit parsing)
    ↓
ui.py (2D structure visualization)
    ↓
molecule.py (property calculations)
    ↓
Display: MW, LogP, TPSA, HBD/HBA, Lipinski rules
```

### Step 2: Database Lookup & Bioactivity
```
RDKit Molecule object
    ↓
pubchem.py (SMILES → InChIKey lookup)
    ↓
pubchem.py (fetch metadata: name, synonyms, CID)
    ↓
chembl.py (InChIKey → ChEMBL ID lookup)
    ↓
chembl.py (ChEMBL ID → bioactivity records)
    ↓
ui.py (display filtered bioactivity results)
```

### Step 3: Similarity Search Pipeline
```
CSV Files (Query & Reference)
    ↓
similarity_search.py (load and validate SMILES)
    ↓
molecule.py (convert SMILES to RDKit molecules)
    ↓
similarity_search.py (compute Morgan fingerprints - radius configurable)
    ↓
similarity_search.py (compute Tanimoto similarity scores)
    ↓
similarity_search.py (rank results by similarity)
    ↓
similarity_search.py (generate structure images and plots)
    ↓
ui.py (display results table, ranking plots, download CSV)
```

## Testing Architecture

### Test Suite: 452 Tests, Comprehensive Coverage (76.05% Overall)

**QSAR Module Coverage (106 tests):****
- `app/qsar/train_models.py`: **100% ✅** (15 tests)
- `app/qsar/model_visualizations.py`: **85.71% ✅** (20 tests)
- `app/qsar/explain.py`: **100% ✅** (6 tests)
- `app/qsar/visualize.py`: **100% ✅** (7 tests)
- `app/qsar/qsar_prediction.py`: **100% ✅** (15 tests)
- `app/qsar/predict.py`: **100% ✅** (8 tests)
- `app/qsar/features.py`: **100% ✅** (8 tests)
- `app/qsar/preprocessing.py`: **100% ✅** (11 tests)
- `app/qsar/data_loader.py`: **100% ✅** (7 tests)

**Similarity Search Module Coverage (197 tests):**
- `app/similarity_search/cli.py`: **100% ✅** (10 tests)
- `app/similarity_search/fingerprints.py`: **100% ✅** (11 tests) 
- `app/similarity_search/pipeline.py`: **100% ✅** (49 tests)
- `app/similarity_search/validators.py`: **100% ✅** (12 tests)
- `app/similarity_search/visualization.py`: **96.51% ✅** (22 tests) - 3 uncovered logging lines
- CLI tests: 40 basic + 11 visualization + 44 edge cases = 95 tests

**Core Module Coverage (100%):**
- `app/chembl.py`: 100% ✅ (17 tests)
- `app/molecule.py`: 100% ✅ (16 tests)
- `app/pubchem.py`: 100% ✅ (13 tests)
- `app/config.py`: 100% ✅ 
- `app/utils.py`: 100% ✅ (8 tests)
- `app/validators.py`: 100% ✅ (14 tests)
- Additional core module tests (82 tests)

**Overall Project Coverage: 76.18%**
**Total Tests: 449 (all passing ✅)**

**Similarity Search Tests (57 tests across 3 files, organized by category):**
- `test_similarity_search_cli_basic.py` - CLI interface & result processing (10 tests)
- `test_similarity_search_cli_visualization.py` - CLI visualization generation (19 tests)
- `test_similarity_search_cli_edge_cases.py` - Edge cases, exception paths, error handling (28 tests)

**Testing Strategy:**
- Unit tests with mocked API calls (no live API dependencies)
- Exception handling and edge case coverage for all three target modules
- Pipeline integration tests with various molecule counts and configurations
- Streamlit cache handling via conftest.py
- Comprehensive similarity search tests covering:
  - Morgan fingerprint computation with various radii
  - Tanimoto similarity calculations
  - Batch processing and ranking
  - Structure image generation with caching
  - CSV export with proper column ordering
  - Visualization and plotting with error recovery
  - CLI argument parsing with proper exit codes
  - ValueError, FileNotFoundError, and generic exception handling
  - Empty dataset and edge case detection
  - Invalid SMILES and molecule pair handling

**Running Tests:**
```bash
uv run pytest tests/ -v                           # Run all 452 tests
uv run pytest tests/ --cov=app --cov-report=html # With coverage report (76%)

# QSAR module tests (106 tests)
uv run pytest tests/test_qsar_*.py -v             # All QSAR tests
uv run pytest tests/test_qsar_train_models.py -v  # Model training
uv run pytest tests/test_qsar_model_visualizations.py -v  # Visualizations

# Similarity search tests (197 tests)
uv run pytest tests/test_similarity_search_*.py -v  # All similarity search
uv run pytest tests/test_similarity_search_fingerprints.py -v   # Fingerprints
uv run pytest tests/test_similarity_search_pipeline.py -v       # Pipeline
uv run pytest tests/test_similarity_search_validators.py -v     # Validators
uv run pytest tests/test_similarity_search_visualization.py -v  # Visualization
uv run pytest tests/test_similarity_search_cli_*.py -v          # CLI tests (95)

# Core module tests (150 tests)
uv run pytest tests/test_chembl.py -v             # ChEMBL integration
uv run pytest tests/test_molecule.py -v           # Molecular operations  
uv run pytest tests/test_pubchem.py -v            # PubChem integration
uv run pytest tests/test_validators.py -v         # Input validation
```

## CI/CD Pipeline

**GitHub Actions Workflow** (`.github/workflows/tests.yml`)
- Triggers on: Push to `main` or `dev` branches, Pull requests
- Python versions: 3.11 and 3.12
- Steps:
  1. Install dependencies via uv
  2. Run Ruff linting and format checks
  3. Execute pytest with coverage reporting
  4. Upload coverage to Codecov (optional)

## API Integration

### Caching Strategy (New - March 2026)

**Streamlit Cache Layer** with 24-hour TTL (86400 seconds):
```python
@st.cache_data(ttl=86400)
def get_pubchem_metadata(mol) -> dict
```

**Cached Functions:**
- `pubchem.get_pubchem_metadata()` - PubChem lookups
- `chembl.get_chembl_molecule()` - ChEMBL molecule resolution
- `chembl.get_chembl_bioactivity()` - Bioactivity data retrieval

**Benefits:**
- Eliminates duplicate API calls for same molecule within 24 hours
- Reduces external API quota usage
- Improves user experience with instant results for repeated queries
- Gracefully expires data after 24 hours for freshness

**Structure Image Caching:**
- `create_structure_image()` uses `@functools.lru_cache(maxsize=1024)`
- Caches up to 1024 unique molecule pair images
- Critical for large similarity search result sets

### PubChem (pubchempy library)
- Query compounds by SMILES
- Retrieve: IUPAC name, synonyms, molecular formula, weight, CID
- Error handling for invalid or not-found compounds

### ChEMBL (REST API)
- Query molecules by InChIKey
- Retrieve: molecule metadata, pref_name, molecule_type, max_phase
- Query bioactivity: target name, standard type (IC50, EC50, Ki), activity values, units
- Confidence scoring for data quality

## Similarity Search Algorithm

### Morgan Fingerprints (ECFP)
**Extended Connectivity Fingerprints** encode molecular structure as 2048-bit binary vectors:
- **Radius**: Determines how many bonds away to consider neighbors (0-5)
  - Radius 0: Only atomic properties (element, charge, etc.)
  - Radius 1: Direct neighbors (1 bond away)
  - Radius 2: Extended neighbors (2 bonds away) - **ECFP4 standard (recommended)**
  - Radius 3+: Extended neighborhoods (more specific/sensitive)
- Each bit position represents a unique structural feature
- Identical molecules produce identical fingerprints

### Tanimoto Similarity
**Industry-standard metric** for comparing molecular structures:
```
Tanimoto = (bits in common) / (bits in either molecule)
Range: 0.0 (completely different) to 1.0 (identical)
```

**Key Characteristics:**
- Symmetric: Similarity(A, B) = Similarity(B, A)
- Efficient: O(1) computation for fixed-size vectors
- Interpretable: Direct measure of structural overlap

### Batch Processing Pipeline
1. **Load Data**: Read CSV files with SMILES strings
2. **Validate**: Check SMILES syntax and convert to RDKit molecules
3. **Fingerprint**: Compute Morgan fingerprints for all molecules
4. **Compare**: Compute pairwise Tanimoto similarities
5. **Rank**: Sort by similarity (descending) and filter top N hits
6. **Visualize**: Generate comparison images and ranking plots
7. **Export**: Output results as CSV with similarity scores

**In-Memory Architecture:**
- No disk I/O for intermediate results
- All processing in RAM
- Suitable for datasets up to thousands of molecules
- Streamlit caching for repeated comparisons

## Dependencies

**Core Runtime:**
- `rdkit` - Molecular chemistry toolkit
- `streamlit` - Web UI framework
- `python-dotenv` - Environment variables
- `pubchempy` - PubChem API client
- `requests` - HTTP library
- `matplotlib` - Plotting and visualization for similarity search results

**Development:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `ruff` - Linting and formatting
- `prek` - Pre-commit hooks

---

## Project Status Summary

### ✅ Completed Features

**Core Application (Single Molecule Analysis)**
- Molecular structure visualization
- Property calculations (6 descriptors)
- Lipinski's Rule-of-5 evaluation
- PubChem metadata retrieval
- ChEMBL bioactivity lookup
- Bioactivity filtering and sorting
- Comprehensive test coverage: 6/6 modules at 100%

**QSAR Module** ✨
- Machine learning model training (Random Forest + XGBoost)
- Real ChEMBL EGFR IC50 bioactivity data
- Feature computation (Morgan fingerprints + RDKit descriptors)
- SHAP-based model explainability
- Performance visualization (residuals, calibration, feature importance)
- Comprehensive test coverage: 106 tests, 95% average coverage

**Similarity Search Module**
- Morgan fingerprint computation (flexible radius)
- Tanimoto similarity metric
- Batch processing pipeline
- Multi-query molecular search
- Structure image generation (side-by-side)
- Similarity ranking plots with matplotlib
- CSV import/export functionality
- CLI interface for command-line usage
- Comprehensive test coverage: 197 tests, 99% average coverage

**Development Infrastructure**
- GitHub Actions CI/CD pipeline
- Automated testing on every push
- Coverage reporting
- Ruff linting and formatting
- Pre-commit hooks (prek)

### Test Coverage

- **Total Tests**: 449 passing ✅
- **Overall Coverage**: 76.18%
- **Core Modules**: 100% (chembl, molecule, pubchem, utils, validators, config)
- **Similarity Search**: 197 tests, 99% average coverage
- **QSAR Module**: 106 tests, 95% average coverage
- **Known Gap**: UI module (app/ui.py - Streamlit components typically not unit tested)

### Architecture Highlights

- **In-Memory Processing**: All similarity search operations happen in RAM
- **No External Storage**: No database or file I/O during processing
- **Stateless API Design**: Each function is independent and testable
- **Mocked External APIs**: Tests don't depend on live API access
- **Modular Design**: Each module has a single responsibility
- **Comprehensive Error Handling**: All exception paths covered or intentionally excluded
- **Smart Caching**: 24-hour TTL on PubChem/ChEMBL API calls and structure images
- **Type Safety**: Full type hints throughout for IDE support