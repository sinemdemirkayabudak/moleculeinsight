"""Input validation utilities."""

from rdkit import Chem


def validate_smiles(smiles: str) -> tuple[bool, str]:
    """
    Validate SMILES string using RDKit.

    Performs both format validation and structural validation:
    1. Checks for empty input and length limits
    2. Validates against known invalid characters
    3. Attempts to parse into valid RDKit molecule object

    Parameters
    ----------
    smiles : str
        SMILES string to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message) - error_message is empty string if valid
    """
    if not smiles or not smiles.strip():
        return False, "Empty SMILES input"

    if len(smiles) > 300:
        return False, "SMILES string too long (max 300 chars)"

    # Check for invalid characters first (fast validation)
    # Allowed: atoms (C,N,O,P,S,F,Cl,Br,I,H), bonds (@,+,-,=,#,~), rings ([],(),0-9,%), stereochemistry (/,\,:), other (.,*)
    allowed_chars = "CNOPSFClBrIHcnops@+=-#~[]()0123456789%/\\:.*"
    if not all(c in allowed_chars for c in smiles):
        return False, "Invalid characters in SMILES"

    # Deep structural validation with RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES structure"
        return True, ""
    except Exception as e:
        return False, f"SMILES parsing error: {str(e)}"
