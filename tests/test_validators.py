"""Tests for input validation functions."""

from app.validators import validate_smiles


class TestValidateSmiles:
    """Test suite for SMILES validation."""

    def test_valid_smiles_benzene(self):
        """Test valid benzene SMILES."""
        is_valid, msg = validate_smiles("c1ccccc1")
        assert is_valid is True
        assert msg == ""

    def test_valid_smiles_aspirin(self):
        """Test valid aspirin SMILES (complex molecule)."""
        is_valid, msg = validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert is_valid is True
        assert msg == ""

    def test_valid_smiles_with_rings(self):
        """Test SMILES with ring notation."""
        is_valid, msg = validate_smiles("C1CCCCC1")
        assert is_valid is True

    def test_valid_smiles_with_charges(self):
        """Test SMILES with charged atoms."""
        is_valid, msg = validate_smiles("[NH4+]")
        assert is_valid is True

    def test_empty_smiles(self):
        """Test empty SMILES string."""
        is_valid, msg = validate_smiles("")
        assert is_valid is False
        assert "Empty" in msg

    def test_whitespace_only(self):
        """Test whitespace-only SMILES."""
        is_valid, msg = validate_smiles("   ")
        assert is_valid is False
        assert "Empty" in msg

    def test_too_long_smiles(self):
        """Test SMILES exceeding 300 character limit."""
        long_smiles = "C" * 301
        is_valid, msg = validate_smiles(long_smiles)
        assert is_valid is False
        assert "too long" in msg

    def test_smiles_at_limit(self):
        """Test SMILES exactly at 300 character limit."""
        max_smiles = "C" * 300
        is_valid, msg = validate_smiles(max_smiles)
        assert is_valid is True

    def test_invalid_characters(self):
        """Test SMILES with invalid characters."""
        is_valid, msg = validate_smiles("c1ccccc1XYZ")
        assert is_valid is False
        assert "Invalid characters" in msg

    def test_invalid_special_chars(self):
        """Test SMILES with disallowed special characters."""
        is_valid, msg = validate_smiles("c1ccccc1@#$")
        assert is_valid is False

    def test_valid_stereo_notation(self):
        """Test valid stereochemistry notation."""
        is_valid, msg = validate_smiles("[C@H]")
        assert is_valid is True

    def test_valid_bracket_notation(self):
        """Test valid bracket (explicit H) notation."""
        is_valid, msg = validate_smiles("[CH4]")
        assert is_valid is True

    def test_valid_multidigit_rings(self):
        """Test valid multi-digit ring closures."""
        is_valid, msg = validate_smiles("C%10CCCCCCCCC%10")
        assert is_valid is True

    def test_no_spaces_allowed(self):
        """Test that spaces are not allowed in SMILES."""
        is_valid, msg = validate_smiles("c1 ccccc1")
        assert is_valid is False
