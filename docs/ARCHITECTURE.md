# MoleculeInsight Architecture

**For Users:** See [README.md](../README.md) for features and installation.  
**For Developers:** This covers module design, algorithms, and testing strategy.

## Overview

MoleculeInsight combines multiple molecular analysis techniques:
- **Single Molecule**: Lipinski compliance, molecular properties, PubChem/ChEMBL integration
- **Similarity Search**: Morgan fingerprints + Tanimoto similarity for structural matching
- **QSAR Prediction**: XGBoost model for EGFR IC50 bioactivity prediction
- **Virtual Screening**: Batch processing with drug-likeness filtering
- **Scaffold & SAR Explorer** ✨: Murcko scaffolds, activity cliff detection, IC50 matching

## Project Structure (Condensed)

```
app/
├── validators.py                 # SMILES validation
├── molecule.py                   # RDKit operations & Lipinski rules
├── pubchem.py                    # PubChem API (24h cache)
├── chembl.py                     # ChEMBL API (24h cache)
├── utils.py                      # Utilities & safe execution
├── config.py                     # Configuration, logging
├── similarity_search/            # Morgan fingerprints (100% coverage)
│   ├── fingerprints.py           # 2048-bit ECFP4
│   ├── pipeline.py               # Tanimoto similarity
│   ├── cli.py                    # Command-line interface
│   ├── validators.py             # Input validation
│   └── visualization.py          # Structure images & charts
│
├── qsar/                         # EGFR bioactivity (100% coverage)
│   ├── train_models.py           # RF + XGBoost training
│   ├── qsar_prediction.py        # Prediction pipeline
│   ├── features.py               # Morgan + RDKit descriptors
│   ├── explain.py                # SHAP feature importance
│   ├── preprocessing.py          # Data cleaning
│   ├── predict.py, visualize.py, data_loader.py, smiles_processor.py
│   ├── model_visualizations.py   # Generation of model performance plots
│   └── saved_models/             # Trained RF & XGBoost models
│   └── visualizations/           # Model performance plots
│
├── scaffold_sar.py               # Scaffold extraction (99.49% coverage)
├── virtual_screening.py          # Batch QSAR (100% coverage)
├── ui.py                         # Streamlit dashboards (0% - UI code excluded)
└── data/sample/                  # Sample data for demos
```

## Test Coverage (708 tests, 69.45%)

| Module | Coverage | Type |
|--------|----------|------|
| **Single Molecule Analysis** | **100%** | molecule, chembl (100%), pubchem, validators, utils (100%) |
| **QSAR Model** | **99.51%** | train_models 100%, train 100%, model_viz 98.49% |
| **Similarity Search** | **100%** | fingerprints 100%, pipeline 100%, viz 100% |
| **Virtual Screening** | **100%** | 104 lines |
| **Scaffold & SAR Explorer** | **99.49%** | 150 lines |
| **SMILES Processor** | **100%** | 36 lines with comprehensive tests |
| **UI Layer** | **0%** | Streamlit components (excluded by design) |

### Test Distribution
- Single Molecule tests: 100 (100% coverage)
- QSAR tests: 150+ (100% coverage across 9 modules)
- Similarity tests: 200+ (100% coverage)  
- Virtual screening tests: 50+ (100% coverage)
- Scaffold SAR tests: 150+ (99.49% coverage)
- SMILES Processor tests: 36 (100% coverage)
- CLI/Integration tests: 70+ (100% coverage)

Run tests:
```bash
pytest tests/ -v --cov=app --cov-report=html
pytest tests/test_scaffold_sar.py -v         # Scaffold analysis
pytest tests/test_qsar_*.py -v               # QSAR module
pytest tests/test_similarity_*.py -v         # Similarity search
```

## Key Algorithms

### Lipinski's Rule-of-5
Evaluates drug-likeness: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10

### Morgan Fingerprints (ECFP4)
2048-bit binary vectors encoding molecular structure with configurable radius (0-5)

### Tanimoto Similarity
Industry-standard metric: Tanimoto(A, B) = |A ∩ B| / |A ∪ B|
- Range: 0.0 (completely different) to 1.0 (identical)  
- Used for: Molecular similarity, activity cliff detection

### QSAR Models
- **Training**: ~5000 EGFR inhibitors from ChEMBL
- **Features**: 2048 Morgan bits + 8 RDKit descriptors
- **Model**: XGBoost for predictions (CV R² = 0.70)
- **Output**: pIC50 predictions with confidence intervals

### Murcko Scaffolds
Core ring structure extracted using RDKit's `GetScaffoldForMol()`:
- Removes side chains, preserves connectivity
- Groups molecules by structural framework
- Enables SAR analysis by scaffold class

### Activity Cliff Detection
Finds molecule pairs with:
- High structural similarity (Tanimoto > threshold)
- Large activity differences (IC50 ratio > threshold)
- Reveals structure-activity relationships

## External APIs

### PubChem (pubchempy library)
- **Lookup**: Compound by SMILES in namespace
- **Data**: IUPAC name, synonyms, CID
- **Cache**: 24 hours

### ChEMBL (REST API)
- **Queries**: Molecule by InChIKey, bioactivity by molecule_id
- **Data**: Standard value (IC50), units (nM/µM/mM), assay description
- **Cache**: 24 hours

## Dependencies

**Core**: streamlit, rdkit, requests, python-dotenv  
**Dev**: pytest, pytest-cov, pytest-mock, ruff

## Data Flow

```
User Input (SMILES)
    ↓
[Validation] → molecule.py (Lipinski, properties)
    ↓
[Similarity] → fingerprints.py (Morgan) → pipeline.py (Tanimoto)
    ↓
[QSAR] → features.py → train_models.py → predict.py (pIC50)
    ↓
[Scaffold] → scaffold_sar.py (scaffolds, cliffs, IC50)
    ↓
[Results] → ui.py (dashboards, visualizations)
```

## Error Handling Strategy

All modules include comprehensive exception handling:
- **Invalid SMILES**: Return None/empty results, log warning
- **API failures**: Retry with timeout, fallback to cache
- **Missing data**: Graceful degradation, user notification
- **File I/O**: Validate paths, handle missing files
- **Data validation**: Type checking, null handling

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Molecular property calc | <100ms | RDKit local |
| PubChem lookup | 1-2s | API + cache |
| ChEMBL lookup | 1-2s | API + cache |
| Morgan fingerprints (1000 mols) | 1-2s | RDKit batch |
| Tanimoto similarity matrix (100×100) | ~1s | O(n²) |
| Activity cliff detection | 2-5s | Depends on data |
| QSAR prediction | <500ms | Model inference |

## CI/CD Pipeline

**Trigger**: Every push to main/dev  
**Steps**:
1. Ruff linting (no formatting issues)
2. Pytest suite (708 tests, <30 seconds)
3. Coverage report (69.45% overall)

## Module Dependencies

```
validators ←─┐
              ├─→ ui.py (Streamlit dashboard)
molecule     ├─→ utils.py (safe execution)
chembl   ────┤
pubchem  ────┤
         ┌────┘
         │
fingerprints → pipeline.py → visualization.py
         │
         ├── features.py → train_models.py → qsar_prediction.py
         │
         └── scaffold_sar.py (standalone analysis)

virtual_screening.py (uses qsar_prediction + features)
```

## Adding Features

1. **New property**: Add to `molecule.py`, add tests
2. **New API**: Create `new_api.py`, integrate to `chembl.py` pattern
3. **New algorithm**: Create in appropriate module (e.g., `similarity_search/algo.py`)
4. **UI feature**: Add to `ui.py`, ensure testable logic separated
5. **Run tests**: `pytest tests/ --cov=app`

## Key Design Decisions

- **Modular**: Each module has single responsibility
- **Cached**: API responses cached 24 hours to reduce requests
- **Exception-safe**: All functions handle errors gracefully
- **Type-hinted**: Full type annotations for IDE support
- **Tested**: 69.45% coverage (708 tests), 100% on core logic
- **Documented**: Docstrings on all functions and classes
- **Type Safety**: Full type hints throughout for IDE support