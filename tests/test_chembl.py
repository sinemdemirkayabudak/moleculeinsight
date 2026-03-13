"""Tests for ChEMBL API integration."""

from unittest.mock import MagicMock, patch

from rdkit import Chem

from app.chembl import (
    get_chembl_bioactivity,
    get_chembl_molecule,
    get_chembl_target_id,
    get_compound_bioactivity_from_mol,
)


class TestGetChemblTargetId:
    """Test ChEMBL target ID resolution functionality."""

    @patch("app.chembl.get_response_json")
    def test_successful_target_id_resolution(self, mock_get_response):
        """Test successful target ID resolution by name."""
        mock_get_response.return_value = {
            "targets": [
                {
                    "target_chembl_id": "CHEMBL203",
                    "pref_name": "Epidermal growth factor receptor",
                    "target_type": "SINGLE PROTEIN",
                }
            ]
        }

        result = get_chembl_target_id("Epidermal growth factor receptor")

        assert result == "CHEMBL203"
        # Verify the API was called with correct parameters
        mock_get_response.assert_called_once()
        call_args = mock_get_response.call_args
        assert call_args[0][1]["pref_name__iexact"] == "Epidermal growth factor receptor"
        assert call_args[0][1]["limit"] == 1

    @patch("app.chembl.get_response_json")
    def test_no_targets_found(self, mock_get_response):
        """Test handling when target name is not found."""
        mock_get_response.return_value = {"targets": []}

        result = get_chembl_target_id("NonexistentTarget")

        assert result is None

    @patch("app.chembl.get_response_json")
    def test_empty_response_returned(self, mock_get_response):
        """Test handling when API returns empty/None response."""
        mock_get_response.return_value = None

        result = get_chembl_target_id("Epidermal growth factor receptor")

        assert result is None

    @patch("app.chembl.get_response_json")
    def test_missing_target_id_in_response(self, mock_get_response):
        """Test handling when target_chembl_id field is missing."""
        mock_get_response.return_value = {
            "targets": [
                {
                    "pref_name": "Some Target",
                    "target_type": "SINGLE PROTEIN",
                    # target_chembl_id is missing
                }
            ]
        }

        result = get_chembl_target_id("Some Target")

        # Should return None because target_chembl_id is missing
        assert result is None

    @patch("app.chembl.get_response_json")
    def test_api_exception_handling(self, mock_get_response):
        """Test error handling when API raises exception."""
        mock_get_response.side_effect = Exception("Connection timeout")

        result = get_chembl_target_id("Epidermal growth factor receptor")

        assert result is None

    @patch("app.chembl.get_response_json")
    def test_multiple_targets_returns_first(self, mock_get_response):
        """Test that first (most relevant) target is returned when multiple matches."""
        mock_get_response.return_value = {
            "targets": [
                {
                    "target_chembl_id": "CHEMBL203",
                    "pref_name": "Epidermal growth factor receptor",
                },
                {
                    "target_chembl_id": "CHEMBL204",
                    "pref_name": "Epidermal growth factor receptor variant",
                },
            ]
        }

        result = get_chembl_target_id("Epidermal growth factor receptor")

        # Should return the first target ID
        assert result == "CHEMBL203"


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
    """Test ChEMBL bioactivity data retrieval for both molecule and target queries."""

    # Tests for molecule-based queries (backward compatibility)
    @patch("app.chembl.get_response_json")
    def test_molecule_query_success(self, mock_get_response):
        """Test successful bioactivity retrieval for a molecule."""
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

        result = get_chembl_bioactivity(molecule_chembl_id="CHEMBL25")

        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["activities"]) == 1
        assert result["activities"][0]["standard_type"] == "IC50"

    @patch("app.chembl.get_response_json")
    def test_molecule_query_no_results(self, mock_get_response):
        """Test handling when no bioactivity data is found for molecule."""
        mock_get_response.return_value = {"activities": []}

        result = get_chembl_bioactivity(molecule_chembl_id="CHEMBL25")

        assert result["success"] is False
        assert result["count"] == 0
        assert "No bioactivity records found" in result["error"]

    @patch("app.chembl.get_response_json")
    def test_molecule_query_api_error(self, mock_get_response):
        """Test error handling when molecule bioactivity API fails."""
        mock_get_response.return_value = None

        result = get_chembl_bioactivity(molecule_chembl_id="CHEMBL25")

        assert result["success"] is False
        assert "error" in result

    # Tests for target-based queries (new functionality)
    @patch("app.chembl.get_response_json")
    def test_target_query_success(self, mock_get_response):
        """Test successful bioactivity DataFrame retrieval for a target."""
        import pandas as pd

        mock_get_response.return_value = {
            "activities": [
                {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "standard_value": 100.0,
                    "assay_id": "ASSAY001",
                    "document_id": "DOC001",
                },
                {
                    "canonical_smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                    "standard_value": 250.5,
                    "assay_id": "ASSAY002",
                    "document_id": "DOC002",
                },
                {
                    "canonical_smiles": None,  # Missing SMILES - should be filtered
                    "standard_value": 150.0,
                    "assay_id": "ASSAY003",
                    "document_id": "DOC003",
                },
            ]
        }

        result = get_chembl_bioactivity(
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            standard_relation="=",
            standard_units="nM",
        )

        assert result["success"] is True
        assert isinstance(result["data"], pd.DataFrame)
        # Should have 2 records (3 returned, 1 filtered for missing SMILES)
        assert result["count"] == 2
        assert result["total_returned"] == 3
        assert "smiles" in result["data"].columns
        assert "standard_value" in result["data"].columns

    @patch("app.chembl.get_response_json")
    def test_target_query_with_offset(self, mock_get_response):
        """Test that offset parameter is correctly passed for pagination."""
        mock_get_response.return_value = {
            "activities": [
                {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "standard_value": 100.0,
                    "assay_id": "ASSAY001",
                    "document_id": "DOC001",
                }
            ]
        }

        result = get_chembl_bioactivity(
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            limit=10000,
            offset=10000,
        )

        assert result["success"] is True
        # Verify offset was passed to the API
        mock_get_response.assert_called_once()
        call_args = mock_get_response.call_args
        assert call_args[0][1]["offset"] == 10000
        assert call_args[0][1]["limit"] == 10000

    @patch("app.chembl.get_response_json")
    def test_target_query_filters_applied(self, mock_get_response):
        """Test that standard filters are correctly applied in API query."""
        mock_get_response.return_value = {
            "activities": [
                {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "standard_value": 100.0,
                    "assay_id": "ASSAY001",
                    "document_id": "DOC001",
                }
            ]
        }

        result = get_chembl_bioactivity(
            target_chembl_id="CHEMBL203",
            standard_type="IC50",
            standard_relation="=",
            standard_units="nM",
            limit=5000,
        )

        assert result["success"] is True
        # Verify filters were passed to the API
        mock_get_response.assert_called_once()
        call_args = mock_get_response.call_args
        assert call_args[0][1]["standard_type"] == "IC50"
        assert call_args[0][1]["standard_relation"] == "="
        assert call_args[0][1]["standard_units"] == "nM"
        assert call_args[0][1]["limit"] == 5000

    @patch("app.chembl.get_response_json")
    def test_target_query_no_results(self, mock_get_response):
        """Test handling when no results found for target query."""
        mock_get_response.return_value = {"activities": []}

        result = get_chembl_bioactivity(target_chembl_id="CHEMBL203")

        assert result["success"] is False
        assert result["count"] == 0
        assert "No bioactivity records found" in result["error"]

    # Tests for validation (mutual exclusivity and error handling)
    def test_neither_id_provided(self):
        """Test error when neither molecule_chembl_id nor target_chembl_id provided."""
        result = get_chembl_bioactivity()

        assert result["success"] is False
        assert "error" in result

    def test_both_ids_provided(self):
        """Test error when both molecule_chembl_id and target_chembl_id provided."""
        result = get_chembl_bioactivity(
            molecule_chembl_id="CHEMBL25",
            target_chembl_id="CHEMBL203",
        )

        assert result["success"] is False
        assert "error" in result

    @patch("app.chembl.get_response_json")
    def test_invalid_value_filtered(self, mock_get_response):
        """Test that records with None standard_value are filtered out."""
        mock_get_response.return_value = {
            "activities": [
                {
                    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "standard_value": 100.0,
                    "assay_id": "ASSAY001",
                    "document_id": "DOC001",
                },
                {
                    "canonical_smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                    "standard_value": None,  # Invalid - should be filtered
                    "assay_id": "ASSAY002",
                    "document_id": "DOC002",
                },
            ]
        }

        result = get_chembl_bioactivity(target_chembl_id="CHEMBL203")

        assert result["success"] is True
        assert result["count"] == 1  # Only valid record
        assert result["total_returned"] == 2  # Total from API

    @patch("app.chembl.get_response_json")
    def test_api_exception_handling(self, mock_get_response):
        """Test error handling when API raises exception."""
        mock_get_response.side_effect = Exception("Connection timeout")

        result = get_chembl_bioactivity(target_chembl_id="CHEMBL203")

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

    @patch("app.chembl.get_pubchem_metadata")
    @patch("app.chembl.get_chembl_bioactivity")
    @patch("app.chembl.get_chembl_molecule")
    def test_pipeline_with_custom_limit(self, mock_molecule, mock_bioactivity, mock_pubchem):
        """Test pipeline passes custom limit parameter to bioactivity retrieval."""
        mock_molecule.return_value = {
            "success": True,
            "chembl_id": "CHEMBL25",
        }
        mock_bioactivity.return_value = {
            "success": True,
            "count": 50,
            "activities": [{"target_name": f"Target {i}"} for i in range(50)],
        }
        mock_pubchem.return_value = {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "success": True}

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = get_compound_bioactivity_from_mol(mol, limit=50)

        assert result["success"] is True
        assert result["bioactivity"]["count"] == 50
        # Verify the limit was passed to get_chembl_bioactivity with molecule_chembl_id keyword
        mock_bioactivity.assert_called_once_with(molecule_chembl_id="CHEMBL25", limit=50)
