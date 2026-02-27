# MoleculeInsight

A Streamlit web application for analyzing molecular properties and drug-likeness using RDKit and PubChem API integration.

## Overview

**MoleculeInsight** is an interactive tool that helps chemists and researchers quickly analyze molecules by:
- Visualizing molecular structures
- Calculating key molecular properties
- Evaluating **Lipinski's Rule-of-5** compliance for drug-likeness
- Retrieving IUPAC and common names from PubChem

Simply enter a SMILES string (Simplified Molecular Input Line Entry System), and get instant analysis of the molecule.

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
- **PubChem Integration** - Automatically fetches IUPAC and common names
- **Structure Visualization** - 2D molecular structure rendering
- **Input Validation** - Validates SMILES format before processing
- **Caching** - Results cached for 24 hours to improve performance
- **Logging** - Comprehensive logging for debugging

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

The project includes a comprehensive test suite with **47 tests** covering all business logic.

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=app --cov-report=term --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`

### Run Specific Test Files

```bash
# Test input validation
pytest tests/test_validators.py -v

# Test molecular calculations
pytest tests/test_molecule.py -v

# Test PubChem API integration
pytest tests/test_pubchem.py -v

# Test utility functions
pytest tests/test_utils.py -v
```

### Test Coverage

- **test_validators.py** (14 tests) - SMILES validation edge cases
- **test_molecule.py** (11 tests) - RDKit calculations and Lipinski rules
- **test_pubchem.py** (13 tests) - PubChem API and name cleaning
- **test_utils.py** (8 tests) - Safe execution and API response handling

**All tests pass** ✓ with mocking for Streamlit and API calls

## Project Structure

```
moleculeinsight/
├── main.py                      # Entry point - starts Streamlit app
├── app/                         # Main application package
│   ├── __init__.py
│   ├── config.py                # Configuration, logging setup, API endpoints
│   ├── validators.py            # SMILES validation logic
│   ├── utils.py                 # Utility functions (safe_execute, cached API calls)
│   ├── molecule.py              # RDKit molecular operations
│   ├── pubchem.py               # PubChem API integration
│   └── ui.py                    # Streamlit UI and user interface
├── tests/                       # Comprehensive test suite (47 tests)
│   ├── conftest.py              # Pytest configuration & fixtures
│   ├── test_validators.py       # 14 tests for input validation
│   ├── test_molecule.py         # 11 tests for molecular calculations
│   ├── test_pubchem.py          # 13 tests for PubChem API
│   └── test_utils.py            # 8 tests for utility functions
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

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.54.0 | Web app framework |
| `rdkit` | ≥2025.9.3 | Molecular analysis |
| `python-dotenv` | ≥1.2.1 | Environment variables |
| `requests` | ≥2.31.0 | HTTP requests to PubChem API |

### Development Dependencies

- `pytest` ≥9.0.2 - Testing framework
- `pytest-cov` ≥4.1.0 - Code coverage reports
- `pytest-mock` ≥3.12.0 - Mocking for tests
- `ruff` ≥0.15.2 - Code formatter/linter
- `prek` ≥0.3.3 - Pre-commit hooks

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

## Logs

Application logs are written to `moleculeinsight.log`:
- Info: Successful operations
- Warning: API failures (retried with cache)
- Error: Validation or calculation errors
- Exception: Full tracebacks for debugging

View logs:
```bash
tail -f moleculeinsight.log
```

## Development Notes

### Code Quality
- Type hints throughout for IDE support
- Comprehensive error handling with logging
- Input validation before processing
- Cached functions for performance

### Adding New Features
1. Add new properties in `get_rdkit_properties()`
2. Add rules in `lipinski_rules()`
3. Update UI in `main()`
4. Add logging for debugging

## Related Tools

- [RDKit Documentation](https://www.rdkit.org/)
- [PubChem API](https://pubchem.ncbi.nlm.nih.gov/docs/PUG-REST)
- [SMILES Format](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)
- [Lipinski's Rule-of-5](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five)

## License

No license specified. Modify as needed.

## Support

For issues or questions:
1. Check `moleculeinsight.log` for error details
2. Verify SMILES format with RDKit: `python -c "from rdkit import Chem; Chem.MolFromSmiles('...')"`
3. Ensure internet connection for PubChem API calls

---

**Last Updated:** February 27, 2026
