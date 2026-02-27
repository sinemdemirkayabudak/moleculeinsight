# MoleculeInsight Architecture

## File Organization
- `app/config.py` - Configuration and logging setup
- `app/validators.py` - Input validation
- `app/molecule.py` - RDKit molecular operations
- `app/pubchem.py` - PubChem API integration
- `app/ui.py` - Streamlit interface

## Data Flow
1. User enters SMILES
2. `validators.py` validates format
3. `molecule.py` creates RDKit object
4. `pubchem.py` fetches metadata
5. `ui.py` displays results