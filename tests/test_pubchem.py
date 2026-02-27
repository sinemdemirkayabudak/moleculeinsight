"""Tests for PubChem API integration."""

from unittest.mock import patch

from rdkit import Chem

from app.pubchem import _get_pubchem_metadata, get_clean_common_name


class TestGetCleanCommonName:
    """Test common name cleaning function."""

    def test_clean_simple_name(self):
        """Test cleaning a simple chemical name."""
        synonyms = ["Benzene", "Benzol"]
        name = get_clean_common_name(synonyms)
        assert name == "Benzene"

    def test_filters_cas_numbers(self):
        """Test that CAS numbers are filtered out."""
        synonyms = ["71-43-2", "Benzene"]  # CAS number first
        name = get_clean_common_name(synonyms)
        assert name == "Benzene"
        assert "71-43-2" not in name

    def test_filters_pubchem_identifiers(self):
        """Test that PubChem IDs are filtered."""
        synonyms = ["PubChem CID: 241", "Benzene"]
        name = get_clean_common_name(synonyms)
        assert name == "Benzene"

    def test_removes_stereo_descriptors(self):
        """Test removal of stereochemistry notations."""
        synonyms = ["(R)-Lactic acid"]
        name = get_clean_common_name(synonyms)
        assert "(" not in name
        assert ")" not in name

    def test_normalizes_spacing(self):
        """Test that excessive spaces are normalized."""
        synonyms = ["Benzene   with   spaces"]
        name = get_clean_common_name(synonyms)
        # Should have normalized spacing
        assert "   " not in name

    def test_empty_synonyms_returns_unknown(self):
        """Test that empty list returns Unknown."""
        name = get_clean_common_name([])
        assert name == "Unknown"

    def test_all_bad_synonyms_returns_unknown(self):
        """Test that all-bad synonyms returns Unknown."""
        synonyms = ["71-43-2", "PubChem", "12345"]
        name = get_clean_common_name(synonyms)
        assert name == "Unknown"

    def test_mixed_case_title_case(self):
        """Test that output is title-cased."""
        synonyms = ["benzene"]
        name = get_clean_common_name(synonyms)
        assert name[0].isupper()


class TestGetPubchemMetadata:
    """Test PubChem metadata retrieval."""

    def test_none_molecule_returns_unknown(self):
        """Test that None molecule returns Unknown."""
        result = _get_pubchem_metadata(None)
        assert result["iupac"] == "Unknown"
        assert result["common"] == "Unknown"

    @patch("app.pubchem.get_response_json")
    def test_api_failure_returns_unknown(self, mock_get):
        """Test that API failure returns Unknown gracefully."""
        mock_get.return_value = None

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = _get_pubchem_metadata(mol)

        assert result["iupac"] == "Unknown"
        assert result["common"] == "Unknown"

    @patch("app.pubchem.get_response_json")
    def test_no_cid_found_returns_unknown(self, mock_get):
        """Test that missing CID returns Unknown."""
        # First call returns no CID
        mock_get.return_value = {"IdentifierList": {"CID": []}}

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = _get_pubchem_metadata(mol)

        assert result["iupac"] == "Unknown"

    @patch("app.pubchem.get_response_json")
    def test_invalid_cid_returns_unknown(self, mock_get):
        """Test that invalid CID (<=0) returns Unknown."""
        mock_get.return_value = {"IdentifierList": {"CID": [-1]}}

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = _get_pubchem_metadata(mol)

        assert result["iupac"] == "Unknown"

    @patch("app.pubchem.get_response_json")
    def test_successful_metadata_retrieval(self, mock_get):
        """Test successful metadata retrieval."""
        # Mock three API calls
        mock_responses = [
            {"IdentifierList": {"CID": [241]}},  # CID lookup
            {  # IUPAC name
                "PropertyTable": {"Properties": [{"IUPACName": "benzene"}]}
            },
            {  # Synonyms
                "InformationList": {"Information": [{"Synonym": ["Benzene", "Benzol"]}]}
            },
        ]
        mock_get.side_effect = mock_responses

        mol = Chem.MolFromSmiles("c1ccccc1")
        result = _get_pubchem_metadata(mol)

        assert result["iupac"] == "benzene"
        assert result["common"] == "Benzene"
