"""Comprehensive tests for app.qsar.smiles_processor module - 100% coverage.

Tests cover:
- load_egfr_compounds: successful load and error handling
- get_all_smiles: extraction of name->SMILES mapping
- get_cached_bioactivity: retrieval by name and SMILES
- _sanitize_bioactivity: string conversion for PyArrow compatibility
"""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.qsar.smiles_processor import (
    _sanitize_bioactivity,
    get_all_smiles,
    get_cached_bioactivity,
    load_egfr_compounds,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_compounds():
    """Sample compound data for testing."""
    return [
        {
            "name": "Erlotinib",
            "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            "bioactivity": [
                {
                    "target_name": "EGFR",
                    "activity_type": "IC50",
                    "value": 2.1,
                    "units": "nM",
                    "assay_description": "EGFR inhibition assay",
                    "pubmed_id": 12345,
                    "selection_rationale": "Potent EGFR inhibitor",
                    "target_chembl_id": "CHEMBL203",
                }
            ],
        },
        {
            "name": "Gefitinib",
            "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "bioactivity": None,
        },
        {
            "name": "Afatinib",
            "smiles": "CC(C)Nc1c(Cl)cc(cc1Nc2c(I)cc(N3CCOCC3)cc2)NC(=O)C",
        },
    ]


@pytest.fixture
def sample_bioactivity_records():
    """Sample bioactivity records."""
    return [
        {
            "target_chembl_id": "CHEMBL203",
            "target_name": "EGFR",
            "activity_type": "IC50",
            "value": 2.1,
            "units": "nM",
            "assay_description": "EGFR inhibition",
            "pubmed_id": 12345,
            "selection_rationale": "Lead compound",
        },
        {
            "target_chembl_id": "CHEMBL204",
            "target_name": "HER2",
            "activity_type": "IC50",
            "value": 5.4,
            "units": "nM",
            "assay_description": "HER2 inhibition",
            "pubmed_id": None,
            "selection_rationale": None,
        },
    ]


# ============================================================================
# Test load_egfr_compounds
# ============================================================================


class TestLoadEgfrCompounds:
    """Tests for load_egfr_compounds() function."""

    @patch("app.qsar.smiles_processor.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_successful_load(self, mock_file, mock_path, sample_compounds):
        """Test successful loading of EGFR compounds."""
        mock_path.return_value.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=True))
        )

        with patch("app.qsar.smiles_processor.json.load") as mock_json_load:
            mock_json_load.return_value = {"compounds": sample_compounds}
            result = load_egfr_compounds()

        assert len(result) == 3
        assert result[0]["name"] == "Erlotinib"

    @patch("app.qsar.smiles_processor.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_json_decode_error(self, mock_file, mock_path):
        """Test JSONDecodeError handling."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        with patch("app.qsar.smiles_processor.json.load") as mock_json_load:
            mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            with pytest.raises(json.JSONDecodeError):
                load_egfr_compounds()

    @patch("app.qsar.smiles_processor.Path")
    def test_file_not_found_error(self, mock_path):
        """Test FileNotFoundError when data file does not exist."""
        # Create a mock Path instance where exists() returns False
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False

        # Make the Path constructor return our mock instance
        # Also handle the / operator (division) to return the mock
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
        mock_path_instance.parent = mock_path_instance

        with pytest.raises(FileNotFoundError, match="EGFR inhibitors data not found"):
            load_egfr_compounds()


# ============================================================================
# Test get_all_smiles
# ============================================================================


class TestGetAllSmiles:
    """Tests for get_all_smiles() function."""

    def test_extract_smiles_mapping(self, sample_compounds):
        """Test extraction of name->SMILES mapping."""
        result = get_all_smiles(sample_compounds)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["Erlotinib"] == "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"
        assert result["Gefitinib"] == "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"

    def test_empty_compounds_list(self):
        """Test with empty compounds list."""
        result = get_all_smiles([])
        assert result == {}

    def test_single_compound(self):
        """Test with single compound."""
        compounds = [
            {
                "name": "Test",
                "smiles": "CCO",
            }
        ]
        result = get_all_smiles(compounds)
        assert result["Test"] == "CCO"


# ============================================================================
# Test _sanitize_bioactivity
# ============================================================================


class TestSanitizeBioactivity:
    """Tests for _sanitize_bioactivity() function."""

    def test_sanitize_all_values_to_strings(self, sample_bioactivity_records):
        """Test conversion of all values to strings."""
        result = _sanitize_bioactivity(sample_bioactivity_records)

        assert len(result) == 2
        # Check first record
        assert isinstance(result[0]["target_chembl_id"], str)
        assert isinstance(result[0]["target_name"], str)
        assert isinstance(result[0]["value"], str)
        assert result[0]["value"] == "2.1"
        # Check that pubmed_id is converted to string
        assert isinstance(result[0]["pubmed_id"], str)
        assert result[0]["pubmed_id"] == "12345"

    def test_sanitize_none_values(self, sample_bioactivity_records):
        """Test sanitization of None values."""
        result = _sanitize_bioactivity(sample_bioactivity_records)

        # Check second record with None values
        assert result[1]["pubmed_id"] == ""
        assert result[1]["selection_rationale"] == ""

    def test_sanitize_empty_list(self):
        """Test with empty bioactivity list."""
        result = _sanitize_bioactivity([])
        assert result == []

    def test_sanitize_none_input(self):
        """Test with None input."""
        result = _sanitize_bioactivity(None)
        assert result == []

    def test_sanitize_maintains_all_fields(self, sample_bioactivity_records):
        """Test that all fields are preserved after sanitization."""
        result = _sanitize_bioactivity(sample_bioactivity_records)

        record = result[0]
        assert "target_chembl_id" in record
        assert "target_name" in record
        assert "activity_type" in record
        assert "value" in record
        assert "units" in record
        assert "assay_description" in record
        assert "pubmed_id" in record
        assert "selection_rationale" in record


# ============================================================================
# Test get_cached_bioactivity
# ============================================================================


class TestGetCachedBioactivity:
    """Tests for get_cached_bioactivity() function."""

    def test_retrieve_by_compound_name(self, sample_compounds):
        """Test retrieval of bioactivity by compound name."""
        result = get_cached_bioactivity(sample_compounds, "Erlotinib")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["target_name"] == "EGFR"
        assert result[0]["value"] == "2.1"

    def test_retrieve_by_smiles(self, sample_compounds):
        """Test retrieval of bioactivity by SMILES string."""
        smiles = "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"
        result = get_cached_bioactivity(sample_compounds, smiles)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1

    def test_name_case_insensitive(self, sample_compounds):
        """Test that name retrieval is case-insensitive."""
        result = get_cached_bioactivity(sample_compounds, "erlotinib")
        assert result is not None

    def test_compound_not_found(self, sample_compounds):
        """Test with non-existent compound."""
        result = get_cached_bioactivity(sample_compounds, "NonExistent")
        assert result is None

    def test_compound_no_bioactivity(self, sample_compounds):
        """Test with compound that has no bioactivity data."""
        result = get_cached_bioactivity(sample_compounds, "Gefitinib")
        assert result is None

    def test_compound_missing_bioactivity_key(self):
        """Test with compound missing bioactivity key entirely."""
        compounds = [
            {
                "name": "Test",
                "smiles": "CCO",
            }
        ]
        result = get_cached_bioactivity(compounds, "Test")
        assert result is None

    def test_empty_compounds_list(self):
        """Test with empty compounds list."""
        result = get_cached_bioactivity([], "Unknown")
        assert result is None

    def test_bioactivity_sanitized_on_retrieval(self, sample_compounds):
        """Test that retrieved bioactivity is sanitized."""
        result = get_cached_bioactivity(sample_compounds, "Erlotinib")

        assert result is not None
        # Verify all values are strings
        record = result[0]
        assert isinstance(record["value"], str)
        assert isinstance(record["pubmed_id"], str)

    def test_multiple_bioactivity_records(self):
        """Test with compound having multiple bioactivity records."""
        compounds = [
            {
                "name": "MultiTarget",
                "smiles": "CCO",
                "bioactivity": [
                    {
                        "target_name": "EGFR",
                        "activity_type": "IC50",
                        "value": 2.1,
                        "units": "nM",
                        "assay_description": "Test1",
                        "target_chembl_id": "CHEMBL203",
                        "pubmed_id": None,
                        "selection_rationale": None,
                    },
                    {
                        "target_name": "HER2",
                        "activity_type": "IC50",
                        "value": 5.4,
                        "units": "nM",
                        "assay_description": "Test2",
                        "target_chembl_id": "CHEMBL204",
                        "pubmed_id": None,
                        "selection_rationale": None,
                    },
                ],
            }
        ]

        result = get_cached_bioactivity(compounds, "MultiTarget")
        assert len(result) == 2
        assert result[0]["target_name"] == "EGFR"
        assert result[1]["target_name"] == "HER2"
