"""Tests for QSAR data loader functionality."""

from unittest.mock import patch

import pandas as pd

from app.qsar.data_loader import get_egfr_ic50_data, load_egfr_dataset


class TestGetEgfrIc50Data:
    """Test EGFR IC50 data retrieval functionality."""

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_successful_ic50_data_retrieval(self, mock_target_id, mock_bioactivity):
        """Test successful retrieval of EGFR IC50 data."""
        mock_target_id.return_value = "CHEMBL203"
        mock_bioactivity.return_value = {
            "success": True,
            "data": pd.DataFrame(
                {
                    "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],
                    "standard_value": [100.0, 250.5],
                    "assay_id": ["ASSAY001", "ASSAY002"],
                }
            ),
            "count": 2,
            "total_returned": 2,
        }

        result = get_egfr_ic50_data(limit=10000, offset=0)

        assert result["success"] is True
        assert isinstance(result["data"], pd.DataFrame)
        assert result["count"] == 2
        assert result["total_returned"] == 2
        assert len(result["data"]) == 2
        assert "smiles" in result["data"].columns
        assert "standard_value" in result["data"].columns

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_ic50_data_with_pagination(self, mock_target_id, mock_bioactivity):
        """Test that pagination parameters are correctly passed."""
        mock_target_id.return_value = "CHEMBL203"
        mock_bioactivity.return_value = {
            "success": True,
            "data": pd.DataFrame(
                {
                    "smiles": ["SMILES1", "SMILES2"],
                    "standard_value": [100.0, 200.0],
                    "assay_id": ["ASSAY1", "ASSAY2"],
                }
            ),
            "count": 2,
            "total_returned": 2,
        }

        result = get_egfr_ic50_data(limit=1000, offset=5000)

        assert result["success"] is True
        # Verify pagination parameters were passed to bioactivity function
        mock_bioactivity.assert_called_once()
        call_args = mock_bioactivity.call_args
        assert call_args[1]["limit"] == 1000
        assert call_args[1]["offset"] == 5000

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_ic50_data_with_filters(self, mock_target_id, mock_bioactivity):
        """Test that standard filters are correctly applied."""
        mock_target_id.return_value = "CHEMBL203"
        mock_bioactivity.return_value = {
            "success": True,
            "data": pd.DataFrame(
                {
                    "smiles": ["SMILES1"],
                    "standard_value": [100.0],
                    "assay_id": ["ASSAY1"],
                }
            ),
            "count": 1,
            "total_returned": 1,
        }

        result = get_egfr_ic50_data()

        assert result["success"] is True
        # Verify filters were passed to bioactivity function
        mock_bioactivity.assert_called_once()
        call_args = mock_bioactivity.call_args
        assert call_args[1]["standard_type"] == "IC50"
        assert call_args[1]["standard_relation"] == "="
        assert call_args[1]["standard_units"] == "nM"

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_target_id_resolution_failure(self, mock_target_id, mock_bioactivity):
        """Test error handling when target ID resolution fails."""
        mock_target_id.return_value = None  # Target ID not found

        result = get_egfr_ic50_data()

        assert result["success"] is False
        assert "Could not resolve target ID" in result["error"]
        # Bioactivity should not be called if target ID resolution fails
        mock_bioactivity.assert_not_called()

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_bioactivity_query_failure(self, mock_target_id, mock_bioactivity):
        """Test error handling when bioactivity query fails."""
        mock_target_id.return_value = "CHEMBL203"
        mock_bioactivity.return_value = {
            "success": False,
            "error": "No bioactivity records found",
        }

        result = get_egfr_ic50_data()

        assert result["success"] is False
        assert "No bioactivity records found" in result["error"]

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_exception_handling(self, mock_target_id, mock_bioactivity):
        """Test exception handling for unexpected errors."""
        mock_target_id.side_effect = Exception("Network error")

        result = get_egfr_ic50_data()

        assert result["success"] is False
        assert "Network error" in result["error"]

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_empty_dataset_returned(self, mock_target_id, mock_bioactivity):
        """Test handling when API returns empty dataset."""
        mock_target_id.return_value = "CHEMBL203"
        mock_bioactivity.return_value = {
            "success": False,
            "error": "No bioactivity records found",
        }

        result = get_egfr_ic50_data()

        assert result["success"] is False
        assert "error" in result

    @patch("app.qsar.data_loader.get_chembl_bioactivity")
    @patch("app.qsar.data_loader.get_chembl_target_id")
    def test_large_dataset_handling(self, mock_target_id, mock_bioactivity):
        """Test handling of large datasets from API."""
        mock_target_id.return_value = "CHEMBL203"
        # Create a large mock DataFrame
        large_data = pd.DataFrame(
            {
                "smiles": [f"SMILES{i}" for i in range(1000)],
                "standard_value": [float(100 + i) for i in range(1000)],
                "assay_id": [f"ASSAY{i}" for i in range(1000)],
            }
        )
        mock_bioactivity.return_value = {
            "success": True,
            "data": large_data,
            "count": 1000,
            "total_returned": 1000,
        }

        result = get_egfr_ic50_data(limit=10000)

        assert result["success"] is True
        assert result["count"] == 1000
        assert len(result["data"]) == 1000


class TestLoadEgfrDataset:
    """Test EGFR dataset loading wrapper functionality."""

    @patch("app.qsar.data_loader.get_egfr_ic50_data")
    def test_successful_dataset_load(self, mock_get_data):
        """Test successful dataset loading."""
        mock_df = pd.DataFrame(
            {
                "smiles": ["SMILES1", "SMILES2"],
                "standard_value": [100.0, 200.0],
                "assay_id": ["ASSAY1", "ASSAY2"],
            }
        )
        mock_get_data.return_value = {
            "success": True,
            "data": mock_df,
            "count": 2,
            "total_returned": 2,
        }

        result = load_egfr_dataset(limit=10000)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["smiles", "standard_value", "assay_id"]

    @patch("app.qsar.data_loader.get_egfr_ic50_data")
    def test_dataset_load_failure(self, mock_get_data):
        """Test dataset loading when data fetch fails."""
        mock_get_data.return_value = {
            "success": False,
            "error": "No bioactivity records found",
        }

        result = load_egfr_dataset()

        assert result is None

    @patch("app.qsar.data_loader.get_egfr_ic50_data")
    def test_dataset_load_none_response(self, mock_get_data):
        """Test dataset loading when get_egfr_ic50_data returns None."""
        mock_get_data.return_value = None

        result = load_egfr_dataset()

        assert result is None

    @patch("app.qsar.data_loader.get_egfr_ic50_data")
    def test_dataset_load_with_custom_limit(self, mock_get_data):
        """Test dataset loading with custom limit parameter."""
        mock_df = pd.DataFrame(
            {
                "smiles": ["SMILES1"],
                "standard_value": [100.0],
                "assay_id": ["ASSAY1"],
            }
        )
        mock_get_data.return_value = {
            "success": True,
            "data": mock_df,
            "count": 1,
            "total_returned": 1,
        }

        result = load_egfr_dataset(limit=5000)

        assert isinstance(result, pd.DataFrame)
        # Verify custom limit was passed to get_egfr_ic50_data
        mock_get_data.assert_called_once_with(limit=5000)

    @patch("app.qsar.data_loader.get_egfr_ic50_data")
    def test_dataset_load_missing_success_key(self, mock_get_data):
        """Test handling when result is missing 'success' key."""
        mock_get_data.return_value = {
            "data": pd.DataFrame(),
            # 'success' key is missing
        }

        result = load_egfr_dataset()

        assert result is None

    @patch("app.qsar.data_loader.get_egfr_ic50_data")
    def test_dataset_load_missing_data_key(self, mock_get_data):
        """Test handling when result has success=True but missing 'data'."""
        mock_get_data.return_value = {
            "success": True,
            # 'data' key is missing
        }

        result = load_egfr_dataset()

        assert result is None
