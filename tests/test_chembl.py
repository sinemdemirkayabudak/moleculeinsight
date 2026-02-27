"""Tests for ChEMBL API integration."""

from unittest.mock import MagicMock, patch

from rdkit import Chem

from app.chembl import (
    get_chembl_bioactivity,
    get_chembl_molecule,
    get_compound_bioactivity_from_mol,
)


class TestGetChemblMolecule:
    """Test ChEMBL molecule lookup functionality."""

    @patch("app.chembl.get_response_json")
    def test_successful_molecule_lookup(self, mock_get_response):
        """Test successful ChEMBL molecule lookup by InChIKey."""
        mock_get_response.return_value = {
            "molecules": [
                {
                    "molecule_chembl_id": "CHEMBL25",
                    "pref_name": "Aspirin",
                    "molecule_type": "Small molecule",
                    "max_phase": 4,
                }
            ]
        }

        result = get_chembl_molecule("InChIKey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N")

        assert result["success"] is True
        assert result["chembl_id"] == "CHEMBL25"
        assert result["pref_name"] == "Aspirin"

    @patch("app.chembl.get_response_json")
    def test_no_molecules_found(self, mock_get_response):
        """Test handling when no molecules are found."""
        mock_get_response.return_value = {"molecules": []}

        result = get_chembl_molecule("InvalidInChIKey")

        assert result["success"] is False
        assert "No ChEMBL match found" in result["message"]

    @patch("app.chembl.get_response_json")
    def test_empty_response(self, mock_get_response):
        """Test handling of empty API response."""
        mock_get_response.return_value = None

        result = get_chembl_molecule("InChIKey=TEST")

        assert result["success"] is False
        assert "No ChEMBL match found" in result["message"]

    @patch("app.chembl.get_response_json")
    def test_api_error_handling(self, mock_get_response):
        """Test error handling when API raises exception."""
        mock_get_response.side_effect = Exception("Connection timeout")

        result = get_chembl_molecule("InChIKey=TEST")

        assert result["success"] is False
        assert "error" in result


class TestGetChemblBioactivity:
    """Test ChEMBL bioactivity data retrieval."""

    @patch("app.chembl.get_response_json")
    def test_successful_bioactivity_retrieval(self, mock_get_response):
        """Test successful bioactivity data retrieval."""
        mock_get_response.return_value = {
            "activities": [
                {
                    "target_chembl_id": "CHEMBL204",
                    "target_pref_name": "Cyclooxygenase-2",
                    "standard_type": "IC50",
                    "standard_value": 0.05,
                    "standard_units": "nM",
                    "assay_description": "COX-2 inhibition assay",
                }
            ]
        }

        result = get_chembl_bioactivity("CHEMBL25")

        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["activities"]) == 1
        assert result["activities"][0]["standard_type"] == "IC50"

    @patch("app.chembl.get_response_json")
    def test_no_bioactivity_data(self, mock_get_response):
        """Test handling when no bioactivity data is found."""
        mock_get_response.return_value = {"activities": []}

        result = get_chembl_bioactivity("CHEMBL25")

        assert result["success"] is True
        assert result["count"] == 0
        assert len(result["activities"]) == 0

    @patch("app.chembl.get_response_json")
    def test_bioactivity_with_limit(self, mock_get_response):
        """Test bioactivity retrieval respects limit parameter."""
        mock_get_response.return_value = {
            "activities": [
                {
                    "target_chembl_id": f"CHEMBL{i}",
                    "target_pref_name": f"Target {i}",
                    "standard_type": "IC50",
                    "standard_value": 100 + i,
                    "standard_units": "nM",
                    "assay_description": f"Test {i}",
                }
                for i in range(20)
            ]
        }

        result = get_chembl_bioactivity("CHEMBL25", limit=20)

        # Verify the limit was passed to the API
        mock_get_response.assert_called_once()
        call_args = mock_get_response.call_args
        assert call_args[0][1]["limit"] == 20

        # Verify bioactivity returns the correct number of activities
        assert result["success"] is True
        assert result["count"] == 20
        assert len(result["activities"]) == 20

    @patch("app.chembl.get_response_json")
    def test_bioactivity_data_cleaning(self, mock_get_response):
        """Test that bioactivity data is properly cleaned."""
        mock_get_response.return_value = {
            "activities": [
                {
                    "target_chembl_id": "CHEMBL204",
                    "target_pref_name": "Cyclooxygenase-2",
                    "standard_type": "IC50",
                    "standard_value": 100.5,
                    "standard_units": "nM",
                    "assay_description": "Test assay",
                    "extra_field": "ignore_this",  # Extra fields should be ignored
                }
            ]
        }

        result = get_chembl_bioactivity("CHEMBL25")

        activity = result["activities"][0]
        assert "extra_field" not in activity
        assert activity["standard_value"] == 100.5

    @patch("app.chembl.get_response_json")
    def test_api_error_handling_bioactivity(self, mock_get_response):
        """Test error handling when bioactivity API fails."""
        mock_get_response.side_effect = Exception("API error")

        result = get_chembl_bioactivity("CHEMBL25")

        assert result["success"] is False
        assert "error" in result


class TestGetCompoundBioactivityFromMol:
    """Test the complete bioactivity pipeline."""

    @patch("app.chembl.get_pubchem_metadata")
    @patch("app.chembl.get_chembl_bioactivity")
    @patch("app.chembl.get_chembl_molecule")
    def test_complete_pipeline_success(self, mock_molecule, mock_bioactivity, mock_pubchem):
        """Test successful execution of complete pipeline."""
        mock_molecule.return_value = {
            "success": True,
            "chembl_id": "CHEMBL25",
        }
        mock_bioactivity.return_value = {
            "success": True,
            "count": 2,
            "activities": [
                {
                    "target_name": "Target 1",
                    "standard_type": "IC50",
                    "standard_value": 100,
                    "standard_units": "nM",
                    "assay_description": "Test 1",
                }
            ],
        }
        mock_pubchem.return_value = {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "success": True}

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = get_compound_bioactivity_from_mol(mol)

        assert result["success"] is True
        assert result["bioactivity"]["count"] == 2

    @patch("app.chembl.get_chembl_molecule")
    def test_pipeline_fails_at_molecule_lookup(self, mock_molecule):
        """Test pipeline failure when molecule lookup fails."""
        mock_molecule.return_value = {"success": False}

        mol = MagicMock()
        result = get_compound_bioactivity_from_mol(mol)

        assert result["success"] is False

    @patch("app.chembl.get_pubchem_metadata")
    @patch("app.chembl.get_chembl_bioactivity")
    @patch("app.chembl.get_chembl_molecule")
    def test_pipeline_exception_handling(self, mock_molecule, mock_bioactivity, mock_pubchem):
        """Test pipeline exception handling."""
        mock_pubchem.side_effect = Exception("PubChem error")

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = get_compound_bioactivity_from_mol(mol)

        assert result["success"] is False
        assert "error" in result

    @patch("app.chembl.get_pubchem_metadata")
    @patch("app.chembl.get_chembl_bioactivity")
    @patch("app.chembl.get_chembl_molecule")
    def test_pipeline_chembl_lookup_failure(self, mock_molecule, mock_bioactivity, mock_pubchem):
        """Test pipeline when ChEMBL lookup fails."""
        mock_pubchem.return_value = {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "success": True}
        mock_molecule.return_value = {"success": False}

        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = get_compound_bioactivity_from_mol(mol)

        assert result["success"] is False
        assert result["stage"] == "chembl_lookup"
