# MoleculeInsight

A Streamlit web application for analyzing molecular properties, drug-likeness, and bioactivity using RDKit, PubChem, and ChEMBL API integration.

## Overview

**MoleculeInsight** is an interactive tool that helps chemists and researchers quickly analyze molecules by:
- 🎨 **Visualizing** molecular structures (2D rendering)
- 📊 **Calculating** key molecular properties (MW, LogP, TPSA, H-bonds, etc.)
- ✓ **Evaluating** Lipinski's Rule-of-5 compliance for drug-likeness
- 🔬 **Retrieving** metadata from PubChem (IUPAC name, synonyms, CID)
- 🎯 **Querying** ChEMBL database for known bioactivity data
- 📈 **Filtering** bioactivity results by activity type, units, and confidence

Simply enter a SMILES string (Simplified Molecular Input Line Entry System), and get instant comprehensive analysis of the molecule.

## Features

### 📊 Calculated Properties
- **Molecular Weight (MW)** - Mass of the molecule
- **LogP** - Lipophilicity (octanol-water partition coefficient)
- **TPSA** - Topological Polar Surface Area
- **H-bond Donors (HBD)** - Count of hydrogen bond donor groups
- **H-bond Acceptors (HBA)** - Count of hydrogen bond acceptor groups
- **Rotatable Bonds** - Count of rotatable bonds

### ✓ Lipinski's Rule-of-5 Evaluation
Assesses drug-likeness:
- MW ≤ 500 Da
- LogP ≤ 5
- HBD ≤ 5
- HBA ≤ 10

### 🔬 Additional Features
- **PubChem Integration** - Fetches IUPAC names, synonyms, molecular formulas
- **ChEMBL Bioactivity** - Queries bioactivity data for known drug targets
- **Bioactivity Filtering** - Filter by activity type (IC50, EC50, Ki, etc.), units (nM, µM, mM), and confidence scores
- **Structure Visualization** - 2D molecular structure rendering
- **Input Validation** - SMILES format validation with detailed error messages
- **Caching** - Results cached for 24 hours to improve performance
- **CI/CD Pipeline** - Automated testing on every push with GitHub Actions

## Installation

### Prerequisites
- Python 3.11+
- macOS (configured for both Intel and Apple Silicon)

### Setup

1. **Navigate to the project directory:**
   ```bash
   cd moleculeinsight
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies using UV:**
   ```bash
   uv sync
   ```
   
   Or install manually:
   ```bash
   pip install streamlit>=1.54.0 rdkit>=2025.9.3 python-dotenv>=1.2.1 requests
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env` (already created)
   - Customize API endpoints if needed (default: PubChem public API)
   ```bash
   cp .env.example .env
   ```

## Usage

### Running the Application

```bash
uv run streamlit run main.py
```

Or with standard Python:
```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`

### Example SMILES Strings

Try these common molecules:

1. **Benzene** (aromatic compound)
   ```
   c1ccccc1
   ```

2. **Aspirin** (acetylsalicylic acid)
   ```
   CC(=O)OC1=CC=CC=C1C(=O)O
   ```

3. **Caffeine**
   ```
   CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   ```

4. **Ethanol**
   ```
   CCO
   ```

5. **Ibuprofen**
   ```
   CC(C)Cc1ccc(cc1)C(C)C(=O)O
   ```

### How to Use

1. **Enter SMILES**: Copy a SMILES string into the "Enter SMILES:" input field
2. **View Structure**: See a 2D visualization of the molecule
3. **Check Properties**: Review calculated molecular descriptors
4. **Evaluate Compliance**: Check Lipinski's Rule-of-5 violations
5. **PubChem Data**: View IUPAC and common names (when available)

## Testing

The project includes a comprehensive test suite with **64 tests** covering 100% of core business logic.

### Test Coverage by Module

| Module | Coverage | Tests |
|--------|----------|-------|
| `app/chembl.py` | **100%** ✅ | 17 tests |
| `app/molecule.py` | **100%** ✅ | 16 tests |
| `app/pubchem.py` | **100%** ✅ | 13 tests |
| `app/utils.py` | **100%** ✅ | 8 tests |
| `app/validators.py` | **100%** ✅ | 14 tests |
| `app/config.py` | **100%** ✅ | - |
| **TOTAL COVERAGE** | **66.67%** | **64 tests** |

### Run All Tests

```bash
uv run pytest tests/ -v
```

### Run with Coverage Report

```bash
uv run pytest tests/ --cov=app --cov-report=term-missing --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`

### Run Specific Test Files

```bash
# Test input validation
uv run pytest tests/test_validators.py -v

# Test molecular calculations  
uv run pytest tests/test_molecule.py -v

# Test PubChem API integration
uv run pytest tests/test_pubchem.py -v

# Test ChEMBL API integration
uv run pytest tests/test_chembl.py -v

# Test utility functions
uv run pytest tests/test_utils.py -v
```

### Testing Strategy

- **Mocked API calls** - No live API dependencies, tests run offline
- **Exception handling** - Tests cover success, failure, and edge cases
- **Integration tests** - Pipeline tests verify end-to-end data flow
- **Streamlit patches** - Cache decorators mocked in conftest.py

**All 64 tests pass** ✓

## Project Structure

```
moleculeinsight/
├── main.py                      # Entry point - starts Streamlit app
├── app/                         # Main application package
│   ├── __init__.py
│   ├── config.py                # Configuration, logging setup, API endpoints
│   ├── validators.py            # SMILES validation logic
│   ├── utils.py                 # Utility functions (safe_execute, API calls)
│   ├── molecule.py              # RDKit molecular operations & Lipinski rules
│   ├── pubchem.py               # PubChem API integration
│   ├── chembl.py                # ChEMBL API integration & bioactivity lookup
│   ├── ui.py                    # Streamlit UI and user interface
│   └── components/
│       └── cards.py             # Reusable UI card components
├── tests/                       # Comprehensive test suite (64 tests)
│   ├── conftest.py              # Pytest configuration & Streamlit fixtures
│   ├── test_validators.py       # 14 tests for input validation
│   ├── test_molecule.py         # 16 tests for molecular calculations
│   ├── test_pubchem.py          # 13 tests for PubChem API
│   ├── test_chembl.py           # 17 tests for ChEMBL API & bioactivity
│   ├── test_utils.py            # 8 tests for utility functions
│   └── components/
│       └── test_cards.py        # UI component tests
├── .github/
│   └── workflows/
│       └── tests.yml            # GitHub Actions CI/CD pipeline
├── docs/
│   └── ARCHITECTURE.md          # Detailed architecture documentation
├── pyproject.toml               # Project metadata and dependencies
├── .env                         # Environment variables (git-ignored)
├── .env.example                 # Template for environment variables
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
└── moleculeinsight.log          # Application logs (auto-generated)
```

## Configuration

### Environment Variables

All API endpoints can be customized via `.env`:

```env
# PubChem API Endpoints
PUBCHEM_CID_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON
PUBCHEM_PROP_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/IUPACName/JSON
PUBCHEM_SYN_URL=https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/synonyms/JSON

# Logging Level
LOG_LEVEL=INFO
```

### Ruff Configuration

Code formatting and linting rules are configured in `pyproject.toml`:
- Line length: 100 characters
- Target Python: 3.11+
- Formatting: Double quotes, space indentation

## Dependencies

### Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| `streamlit` | ≥1.54.0 | Web app framework |
| `rdkit` | ≥2025.9.3 | Molecular analysis & structure rendering |
| `python-dotenv` | ≥1.2.1 | Environment variables |
| `pubchempy` | Latest | PubChem API Python client |
| `requests` | ≥2.31.0 | HTTP requests to ChEMBL API |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| `pytest` | ≥9.0.2 | Testing framework |
| `pytest-cov` | ≥7.0.0 | Code coverage reports |
| `pytest-mock` | ≥3.12.0 | Mocking for tests |
| `ruff` | ≥0.15.2 | Code formatter/linter |
| `prek` | ≥0.3.3 | Pre-commit hooks |

## Troubleshooting

### "Invalid SMILES string"
- Check SMILES syntax (OpenSMILES format)
- Ensure proper bracket notation: `[NH3+]`, `[O-]`
- SMILES must be ≤ 300 characters

### "API request failed"
- Check internet connection
- PubChem API may be temporarily unavailable
- Check logs in `moleculeinsight.log`

### "Invalid characters in SMILES"
Valid SMILES characters:
- Atoms: C, N, O, P, S, F, Cl, Br, I, H
- Bonds: @, +, -, =
- Rings: 0-9, %
- Brackets: [ ], ( )

### Caching Issues
- Cache expires after 24 hours
- Clear Streamlit cache: `streamlit cache clear` or use sidebar button

### Performance
- First API call for a molecule takes ~2-3 seconds
- Subsequent calls use cache (instant)
- Molecule structure rendering is cached

## Continuous Integration / Continuous Deployment

The project uses **GitHub Actions** for automated testing on every push.

### CI/CD Pipeline

**File:** `.github/workflows/tests.yml`

**Triggers:**
- Pushes to `main` or `dev` branches
- Pull requests to `main` or `dev`

**Steps:**
1. Check out code
2. Install Python (3.11 and 3.12)
3. Install dependencies via `uv`
4. Run Ruff linting checks
5. Run Ruff format validation
6. Execute pytest suite with coverage
7. Upload coverage to Codecov (optional)

**View Results:**
- Go to GitHub repo → Actions tab
- Click on workflow run to see detailed logs
- Green ✅ = all tests passed
- Red ❌ = tests failed (see logs for details)

## Logs

Application logs are written to `moleculeinsight.log`:
- **Info**: Successful operations, molecule queries
- **Warning**: API timeouts, missing data
- **Error**: Validation failures, calculation errors  
- **Exception**: Full tracebacks for debugging

View logs:
```bash
tail -f moleculeinsight.log
```

## Development Notes

### Code Quality
- **Type hints** throughout for IDE support
- **Comprehensive error handling** with logging
- **Input validation** before processing
- **Cached functions** for performance
- **100% test coverage** for core modules
- **Ruff linting** enforced in CI/CD

### Adding New Features
1. **New molecular properties**: Add to `get_rdkit_properties()` in `molecule.py`
2. **New drug-likeness rules**: Add to `lipinski_rules()` in `molecule.py`
3. **New API integrations**: Create new module (e.g., `app/new_api.py`) with tests
4. **UI components**: Add to `app/ui.py` and create tests in `tests/test_ui.py`
5. **Update tests**: Run `pytest tests/ --cov=app` to verify coverage
6. **Push to GitHub**: Automated tests run via CI/CD

## Related Tools & References

- [RDKit Documentation](https://www.rdkit.org/) - Molecular chemistry toolkit
- [PubChem API](https://pubchem.ncbi.nlm.nih.gov/docs/PUG-REST) - Chemical database REST API
- [ChEMBL API](https://www.ebi.ac.uk/chembl/api/) - Bioactivity database
- [SMILES Format](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) - Molecular notation
- [Lipinski's Rule-of-5](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five) - Drug-likeness criteria
- [GitHub Actions](https://docs.github.com/en/actions) - CI/CD platform

## License

Open source. Free to use and modify.

## Support

For issues or questions:
1. Check `moleculeinsight.log` for error details
2. Verify SMILES format with RDKit: `python -c "from rdkit import Chem; Chem.MolFromSmiles('...')"`
3. Ensure internet connection for PubChem API calls

## API Endpoints

### PubChem (pubchempy library)

```python
import pubchempy as pcp

# Lookup compound by SMILES
compounds = pcp.get_compounds("CCO", namespace="smiles")

# Get properties
compound = compounds[0]
compound.iupac_name        # IUPAC name
compound.synonyms          # List of common names
compound.molecular_formula # Chemical formula
compound.molecular_weight  # Molecular weight
```

### ChEMBL (REST API)

```
GET /molecule.json?molecule_structures__standard_inchi_key={INCHIKEY}
→ Returns: molecule_chembl_id, pref_name, molecule_type, max_phase

GET /activity.json?molecule_chembl_id={ID}&limit=20
→ Returns: target_chembl_id, target_pref_name, standard_type,
          standard_value, standard_units, assay_description
```

## Architecture Documentation

For detailed architecture, module descriptions, and data flow diagrams, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

**Last Updated:** February 28, 2026  
**Python Version:** 3.11+  
**Test Status:** ✅ 64/64 tests passing  
**Coverage:** 66.67% (100% on core modules)
