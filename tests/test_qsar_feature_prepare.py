"""Tests for QSAR feature preparation wrapper function."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from app.qsar.features import prepare_features


class TestPrepareFeatures:
    """Test feature preparation from cleaned datasets."""

    def test_prepare_features_morgan_success(self):
        """Test successful feature preparation with Morgan fingerprints."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "pIC50": [5.2, 3.5, 4.8],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        assert "X" in result
        assert "y" in result
        assert "smiles" in result
        assert "feature_names" in result
        assert "feature_type" in result
        assert result["X"].shape == (3, 2048)
        assert len(result["y"]) == 3
        assert len(result["smiles"]) == 3
        assert len(result["feature_names"]) == 2048
        assert result["feature_type"] == "morgan"

    def test_prepare_features_descriptors_success(self):
        """Test successful feature preparation with descriptors."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "pIC50": [5.2, 3.5, 4.8],
            }
        )
        result = prepare_features(df, feature_type="descriptors")

        assert result["success"] is True
        assert "X" in result
        assert "y" in result
        assert "smiles" in result
        assert "feature_names" in result
        assert "feature_type" in result
        assert result["X"].shape == (3, 8)
        assert len(result["y"]) == 3
        assert len(result["smiles"]) == 3
        assert len(result["feature_names"]) == 8
        assert result["feature_type"] == "descriptors"

    def test_prepare_features_default_feature_type(self):
        """Test prepare_features defaults to morgan."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )
        result = prepare_features(df)

        assert result["success"] is True
        assert result["X"].shape[1] == 2048  # Morgan default

    def test_prepare_features_case_insensitive(self):
        """Test feature_type is case insensitive."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )

        result_lower = prepare_features(df, feature_type="morgan")
        result_upper = prepare_features(df, feature_type="MORGAN")
        result_mixed = prepare_features(df, feature_type="MoRgAn")

        assert result_lower["success"] is True
        assert result_upper["success"] is True
        assert result_mixed["success"] is True

    def test_prepare_features_invalid_feature_type(self):
        """Test prepare_features with invalid feature type."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )
        result = prepare_features(df, feature_type="invalid_type")

        assert result["success"] is False
        assert "error" in result
        assert "Unknown feature type" in result["error"]

    def test_prepare_features_invalid_smiles(self):
        """Test prepare_features with invalid SMILES."""
        df = pd.DataFrame(
            {
                "smiles": ["INVALID_SMILES"],
                "pIC50": [3.5],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is False
        assert "error" in result

    def test_prepare_features_empty_dataframe(self):
        """Test prepare_features with empty dataframe."""
        df = pd.DataFrame(
            {
                "smiles": [],
                "pIC50": [],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is False
        assert "error" in result

    def test_prepare_features_y_matching(self):
        """Test that y values match X length."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "pIC50": [5.2, 3.5, 4.8],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        assert len(result["y"]) == len(result["X"])

    def test_prepare_features_smiles_matching(self):
        """Test that smiles array matches X length."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "pIC50": [5.2, 3.5, 4.8],
            }
        )
        result = prepare_features(df, feature_type="descriptors")

        assert result["success"] is True
        assert len(result["smiles"]) == len(result["X"])

    def test_prepare_features_mixed_valid_invalid(self):
        """Test prepare_features with mix of valid and invalid SMILES."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "INVALID", "CCO"],
                "pIC50": [5.2, 3.5, 4.8],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        assert result["X"].shape[0] == 2  # Only valid molecules
        assert len(result["y"]) == 2
        assert len(result["smiles"]) == 2

    def test_prepare_features_single_molecule(self):
        """Test prepare_features with single molecule."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        assert result["X"].shape == (1, 2048)
        assert result["y"].shape == (1,)
        assert result["y"][0] == 3.5

    def test_prepare_features_large_dataset(self):
        """Test prepare_features with large dataset."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"] * 50 + ["CC(=O)Oc1ccccc1C(=O)O"] * 50,
                "pIC50": np.random.uniform(3, 7, 100),
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        assert result["X"].shape == (100, 2048)
        assert len(result["y"]) == 100

    def test_prepare_features_return_dict_keys(self):
        """Test prepare_features return dict contains all required keys."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        required_keys = {"success", "X", "y", "smiles", "feature_names", "feature_type"}
        assert required_keys.issubset(set(result.keys()))

    def test_prepare_features_pIC50_values_preserved(self):
        """Test that pIC50 values are preserved correctly."""
        pic50_values = [5.2, 3.5, 4.8]
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "pIC50": pic50_values,
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        np.testing.assert_array_equal(result["y"], pic50_values)

    def test_prepare_features_feature_names_valid(self):
        """Test that feature names are valid and match feature count."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )

        # Test Morgan
        result_morgan = prepare_features(df, feature_type="morgan")
        assert len(result_morgan["feature_names"]) == 2048

        # Test Descriptors
        result_desc = prepare_features(df, feature_type="descriptors")
        assert len(result_desc["feature_names"]) == 8

    def test_prepare_features_with_nan_pIC50(self):
        """Test prepare_features handles missing pIC50 values gracefully."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
                "pIC50": [3.5, np.nan],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        # Should still succeed but y will contain NaN
        assert result["success"] is True
        assert len(result["y"]) == 2
        assert np.isnan(result["y"][1])

    def test_prepare_features_both_types_same_molecules(self):
        """Test that both feature types work on same molecules."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "pIC50": [5.2, 3.5],
            }
        )

        result_morgan = prepare_features(df, feature_type="morgan")
        result_descriptors = prepare_features(df, feature_type="descriptors")

        assert result_morgan["success"] is True
        assert result_descriptors["success"] is True
        # Both should have same number of molecules
        assert result_morgan["X"].shape[0] == result_descriptors["X"].shape[0]
        # But different number of features
        assert result_morgan["X"].shape[1] != result_descriptors["X"].shape[1]

    def test_prepare_features_duplicate_molecules(self):
        """Test prepare_features with duplicate molecules."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCO", "CC(=O)Oc1ccccc1C(=O)O"],
                "pIC50": [3.5, 3.5, 5.2],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is True
        assert result["X"].shape[0] == 3
        # First two rows should be identical (same SMILES)
        np.testing.assert_array_equal(result["X"][0], result["X"][1])

    def test_prepare_features_with_failed_feature_computation(self):
        """Test prepare_features when feature computation returns failure."""
        # Create a dataframe with only invalid SMILES
        df = pd.DataFrame(
            {
                "smiles": ["INVALID1", "INVALID2"],
                "pIC50": [3.5, 4.2],
            }
        )
        result = prepare_features(df, feature_type="morgan")

        assert result["success"] is False
        assert "error" in result

    def test_prepare_features_with_exception(self):
        """Test prepare_features with exception from feature computation."""
        # Create a dataframe with valid data but mock an exception
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )

        # Patch the compute_morgan_fingerprints to raise an exception after returning failure
        with patch("app.qsar.features.compute_morgan_fingerprints") as mock_compute:
            mock_compute.return_value = {"success": False, "error": "Test error"}
            result = prepare_features(df, feature_type="morgan")

            assert result["success"] is False
            assert "error" in result

    def test_prepare_features_exception_at_top_level(self):
        """Test prepare_features exception handler at top level."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "pIC50": [3.5],
            }
        )

        # Patch compute_morgan_fingerprints to return result with get that raises exception
        with patch("app.qsar.features.compute_morgan_fingerprints") as mock_compute:
            result_dict = MagicMock()
            result_dict.get.side_effect = RuntimeError("Result access failed")
            mock_compute.return_value = result_dict

            result = prepare_features(df, feature_type="morgan")

            assert result["success"] is False
            assert "error" in result
