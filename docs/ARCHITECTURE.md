# MoleculeInsight Architecture

## Project Structure

```
moleculeinsight/
├── app/                          # Main application code
│   ├── __init__.py
│   ├── config.py                 # Configuration, logging, environment setup
│   ├── validators.py             # SMILES input validation
│   ├── molecule.py               # RDKit molecular operations & Lipinski rules
│   ├── pubchem.py                # PubChem API integration
│   ├── chembl.py                 # ChEMBL API integration & bioactivity lookup
│   ├── utils.py                  # Utility functions (safe execution, API calls)
│   ├── ui.py                     # Streamlit UI components
│   └── components/
│       └── cards.py              # Reusable UI card components
├── tests/                        # Comprehensive test suite (64 tests)
│   ├── conftest.py               # Pytest configuration & fixtures
│   ├── test_validators.py        # Input validation tests
│   ├── test_molecule.py          # Molecular analysis tests
│   ├── test_pubchem.py           # PubChem integration tests
│   ├── test_chembl.py            # ChEMBL integration tests
│   ├── test_utils.py             # Utility function tests
│   └── components/
│       └── test_cards.py         # UI component tests
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

### UI Layer

**`ui.py`**
- Streamlit application layout
- Input forms and molecule visualization
- Results display with metrics and dataframes
- ChEMBL bioactivity filtering interface

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

## Testing Architecture

### Test Suite: 64 Tests, 100% Core Coverage

**Test Coverage by Module:**
- `app/chembl.py`: 100% ✅
- `app/molecule.py`: 100% ✅
- `app/pubchem.py`: 100% ✅
- `app/config.py`: 100% ✅
- `app/utils.py`: 100% ✅
- `app/validators.py`: 100% ✅
- **TOTAL**: 66.67% (UI code excluded)

**Testing Strategy:**
- Unit tests with mocked API calls (no live API dependencies)
- Exception handling and edge case coverage
- Pipeline integration tests
- Streamlit cache handling via conftest.py

**Running Tests:**
```bash
uv run pytest tests/ -v                           # Run all tests
uv run pytest tests/ --cov=app --cov-report=html # With coverage report
uv run pytest tests/test_chembl.py -v             # Run specific module
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

### PubChem (pubchempy library)
- Query compounds by SMILES
- Retrieve: IUPAC name, synonyms, molecular formula, weight, CID
- Error handling for invalid or not-found compounds

### ChEMBL (REST API)
- Query molecules by InChIKey
- Retrieve: molecule metadata, pref_name, molecule_type, max_phase
- Query bioactivity: target name, standard type (IC50, EC50, Ki), activity values, units
- Confidence scoring for data quality

## Dependencies

**Core Runtime:**
- `rdkit` - Molecular chemistry toolkit
- `streamlit` - Web UI framework
- `python-dotenv` - Environment variables
- `pubchempy` - PubChem API client
- `requests` - HTTP library

**Development:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `ruff` - Linting and formatting