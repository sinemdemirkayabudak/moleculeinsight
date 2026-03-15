# MoleculeInsight

> **Users:** Install and use the app with this guide.  
> **Developers:** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module design, algorithms, and testing details.

A Streamlit web application for analyzing molecular properties, drug-likeness, and bioactivity using RDKit, PubChem, and ChEMBL API integration.

## Overview

**MoleculeInsight** is an interactive tool that helps chemists and researchers quickly analyze molecules by:
- **Visualizing** molecular structures (2D rendering)
- **Calculating** key molecular properties (MW, LogP, TPSA, H-bonds, etc.)
- **Evaluating** Lipinski's Rule-of-5 compliance for drug-likeness
- **Retrieving** metadata from PubChem (IUPAC name, synonyms, CID)
- **Querying** ChEMBL database for known bioactivity data
- **Filtering** bioactivity results by activity type, units, and confidence

Simply enter a SMILES string (Simplified Molecular Input Line Entry System), and get instant comprehensive analysis of the molecule.

### Core Features

**Molecular Analysis**
- **Molecular Properties** - MW, LogP, TPSA, HBD, HBA, rotatable bonds, aromaticity
- **Lipinski's Rule-of-5** - Drug-likeness evaluation with violation count
- **PubChem Integration** - IUPAC names, synonyms, molecular formulas
- **ChEMBL Bioactivity** - Bioactivity data and filtering (IC50, EC50, Ki, etc.)

**Similarity Search**
- **Morgan Fingerprints** - ECFP4 with configurable radius (0-5)
- **Tanimoto Similarity** - Industry-standard molecular comparison
- **Batch Processing** - Compare multiple query molecules against reference library
- **CSV Export** - Download results with similarity scores

**QSAR Bioactivity Prediction**
- **ML Models** - Random Forest and XGBoost models trained on EGFR ChEMBL data
- **pIC50 Prediction** - Predict kinase binding affinity
- **SHAP Explainability** - Feature importance visualization
- **Cross-Validation** - 5-fold CV with R²=0.70 performance

**Virtual Screening**
- **Batch QSAR** - Screen multiple molecules in one run
- **CSV Upload** - Process large compound libraries
- **Drug-Likeness Filter** - QED and Lipinski Rule-of-5 compliance
- **Ranked Results** - Output sorted by predicted activity

**Scaffold & SAR Analysis** ✨ NEW
- **Murcko Scaffolds** - Extract scaffold framework from molecules
- **Activity Cliffs** - Detect similar structures with large activity differences
- **Fingerprint Similarity** - Compute Tanimoto similarity for structure comparison
- **IC50 Matching** - Fetch bioactivity data for submitted molecules

## Quick Start

**Prerequisites:** Python 3.11+, macOS (Intel/Apple Silicon)

1. **Install dependencies:**
   ```bash
   cd moleculeinsight
   uv sync                    # or: pip install streamlit rdkit python-dotenv requests
   ```

2. **Run the app:**
   ```bash
   uv run streamlit run main.py
   ```

Open `http://localhost:8501` in your browser.

## Example Molecules & CSV Format

**Sample SMILES:**
```
Benzene:  c1ccccc1
Aspirin:  CC(=O)OC1=CC=CC=C1C(=O)O
Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

**CSV Format:**
```csv
smiles,name
CCO,Ethanol
CC(=O)OC1=CC=CC=C1C(=O)O,Aspirin
```

## Testing & Coverage

**Current Status:** 621 tests, **72.40% overall coverage**

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| **Scaffold & SAR** | **99.50%** ✅ | Core: 150 lines, 1 uncovered branch |
| **Core Analysis** | **100%** ✅ | molecule, chembl, pubchem, validators, utils |
| **QSAR Prediction** | **98.3%** ✅ | 9 modules, train.py 97.59%, model_viz 85.63% |
| **Similarity Search** | **98.7%** ✅ | 5 modules, visualization 96.51% |
| **Virtual Screening** | **100%** ✅ | 104 lines, full coverage |

**Run tests:**
```bash
uv run pytest tests/ -v                           # All 621 tests
uv run pytest tests/ --cov=app --cov-report=html # Generate HTML report
uv run pytest tests/test_scaffold_sar.py -v      # Scaffold SAR tests
```

## Project Structure

```
moleculeinsight/
├── main.py                      # Streamlit app entry point
├── app/
│   ├── config.py, validators.py, utils.py, molecule.py, pubchem.py, chembl.py, ui.py
│   ├── virtual_screening.py     # Batch QSAR pipeline (36 tests, 100%)
│   ├── scaffold_sar.py          # Scaffold extraction & activity cliffs (147 tests, 99.5%)
│   ├── qsar/                    # EGFR bioactivity prediction (102 tests, 98%)
│   │   └── saved_models/        # Trained RF + XGBoost models
│   ├── similarity_search/       # Morgan fingerprints & Tanimoto similarity (197 tests, 97%)
│   ├── components/              # UI card components
│   └── data/sample/             # Sample data for demos
├── tests/                       # 635 tests, 72.4% coverage
│   ├── test_scaffold_sar.py     # 147 tests, 99.5% coverage (NEW)
│   ├── test_qsar_*.py           # QSAR tests (102 tests)
│   ├── test_similarity_*.py     # Similarity search tests (197 tests)
│   └── test_*.py                # Core module tests (82 tests)
├── docs/
│   ├── ARCHITECTURE.md          # Detailed architecture & algorithms
│   └── ANNOTATIONS.md           # Feature annotations (EGFR model)
├── pyproject.toml, .env, .gitignore
└── .github/workflows/tests.yml  # GitHub Actions CI/CD
```

## Configuration

**Environment variables** (`.env`):
```env
PUBCHEM_CID_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON
LOG_LEVEL=INFO
```

**Dependencies:** Streamlit, RDKit, requests, python-dotenv, matplotlib  
**Dev:** pytest, pytest-cov, pytest-mock, ruff

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Invalid SMILES | Check OpenSMILES format, use proper brackets `[NH3+]`, max 300 chars |
| API request failed | Check internet, see `moleculeinsight.log` for errors |
| Cache issues | Run `streamlit cache clear` or use sidebar button |
| Performance | First API call ~2-3s, cached calls instant |

## Development

**Testing:** All tests run via GitHub Actions CI/CD on push  
**Linting:** Ruff enforced in CI/CD (100-char line length)  
**Type hints:** Full annotations throughout codebase  
**Logging:** See `moleculeinsight.log` for application events  

**Adding features:**
1. Implement code with full type hints and docstrings
2. Write tests (aim for 100% coverage)
3. Run `uv run pytest tests/ --cov=app` locally
4. Push to GitHub (CI/CD validates automatically)

## References

**Full documentation:** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed module design, algorithms, and data flow.

**License:** Open source. Free to use and modify.

