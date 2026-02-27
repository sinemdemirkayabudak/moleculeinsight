def validate_smiles(smiles: str) -> tuple[bool, str]:
    """Validate SMILES string format and length."""
    if not smiles or not smiles.strip():
        return False, "Empty SMILES input"

    if len(smiles) > 300:
        return False, "SMILES string too long (max 300 chars)"

    # Check for invalid characters
    # Allowed: atoms (C,N,O,P,S,F,Cl,Br,I,H), bonds (@,+,-,=), rings ([],(),0-9,%), configs
    allowed_chars = "CNOPSFClBrIHcnops@+=-[]()0123456789%"
    if not all(c in allowed_chars for c in smiles):
        return False, "Invalid characters in SMILES"

    return True, ""
