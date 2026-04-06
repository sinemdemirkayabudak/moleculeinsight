# MoleculeInsight

> **Users:** Install and use the app with this guide.  
> **Developers:** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module design, algorithms, and testing details.

A Streamlit web application for analyzing molecular properties, drug-likeness, and bioactivity using RDKit, PubChem, and ChEMBL API integration.

## Overview

**MoleculeInsight** is a comprehensive molecular analysis platform with 5 integrated tools:
- **Single Molecule Analysis** - Visualize structures, calculate properties, evaluate drug-likeness, retrieve PubChem/ChEMBL bioactivity
- **Similarity Search** - Find structurally similar compounds using Morgan fingerprints and Tanimoto similarity
- **QSAR Prediction** - Predict EGFR kinase binding affinity using trained XGBoost QSAR ML model with SHAP explainability
- **Virtual Screening** -  Batch screen molecules through QSAR pipeline, filter by drug-likeness, rank by predicted activity
- **Scaffold & SAR Explorer** - Extract Murcko scaffolds, detect activity cliffs, analyze structure-activity relationships

Enter SMILES strings or upload CSV files for instant multi-faceted molecular analysis.

### Core Features

**Single Molecule Analysis**
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

**Scaffold & SAR Explorer**
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

**Current Status:** 708 tests, **69.45% overall coverage**

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| **Scaffold & SAR Explorer** | **99.49%** ✅ | Core: 148 lines |
| **Single Molecule Analysis** | **100%** ✅ | molecule, pubchem, validators |
| **QSAR Prediction** | **99.51%** ✅ | 9 modules + smiles_processor (100%) |
| **Similarity Search** | **100%** ✅ | 5 modules, visualization 100% |
| **Virtual Screening** | **100%** ✅ | 104 lines, full coverage |

**Run tests:**
```bash
uv run pytest tests/ -v                           # All 708 tests
uv run pytest tests/ --cov=app --cov-report=html # Generate HTML report
uv run pytest tests/test_scaffold_sar.py -v      # Scaffold SAR tests
uv run pytest tests/test_qsar_*.py -v            # QSAR model tests
```

## Project Structure

```
moleculeinsight/
├── main.py                      # Streamlit app entry point
├── app/
│   ├── config.py, validators.py, utils.py, molecule.py, pubchem.py, chembl.py (100%), ui.py
│   ├── virtual_screening.py     # Batch QSAR pipeline (100% coverage)
│   ├── scaffold_sar.py          # Scaffold extraction & activity cliffs (99.49% coverage)
│   ├── qsar/                    # EGFR bioactivity prediction (100% coverage)
│   │   ├── smiles_processor.py  # SMILES utilities (100% coverage)
│   │   ├── model_visualizations.py # Performance plots (98.49% coverage)
│   │   └── saved_models/        # Trained RF + XGBoost models
│   ├── similarity_search/       # Morgan fingerprints & Tanimoto similarity (100% coverage)
│   ├── components/              # UI card components
│   └── data/sample/             # Sample data for demos
├── tests/                       # 708 tests, 69.45% coverage
│   ├── test_scaffold_sar.py     # Tests with 99.49% coverage
│   ├── test_qsar_model_visualizations.py # Model visualization tests
│   ├── test_qsar_smiles_processor.py # SMILES processor tests (100% coverage)
│   ├── test_qsar_*.py           # QSAR tests (100% coverage)
│   ├── test_similarity_*.py     # Similarity search tests (100% coverage)
│   └── test_*.py                # Core module tests (100% coverage)
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

