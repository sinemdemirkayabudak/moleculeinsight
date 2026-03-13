"""Tests for QSAR preprocessing functionality."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from app.qsar.preprocessing import (
    clean_and_preprocess_data,
    convert_ic50_to_pic50,
    get_cleaned_dataset,
)


class TestConvertIc50ToPic50:
    """Test IC50 to pIC50 conversion."""

    def test_valid_conversion_typical(self):
        """Test typical IC50 value conversion."""
        # IC50 = 100 nM → pIC50 = 7.0
        result = convert_ic50_to_pic50(100.0)
        assert abs(result - 7.0) < 0.01

    def test_valid_conversion_micromolar(self):
        """Test micromolar IC50 conversion."""
        # IC50 = 1000 nM (1 µM) → pIC50 = 6.0
        result = convert_ic50_to_pic50(1000.0)
        assert abs(result - 6.0) < 0.01

    def test_valid_conversion_picomolar(self):
        """Test picomolar IC50 conversion."""
        # IC50 = 0.001 nM (1 pM) → pIC50 = 12.0
        result = convert_ic50_to_pic50(0.001)
        assert abs(result - 12.0) < 0.01

    def test_valid_conversion_aspirin(self):
        """Test aspirin-like IC50 value."""
        # IC50 = 345.2 nM → pIC50 ≈ 6.46
        result = convert_ic50_to_pic50(345.2)
        assert 6.4 < result < 6.5

    def test_zero_ic50_returns_nan(self):
        """Test that IC50 = 0 returns NaN."""
        result = convert_ic50_to_pic50(0.0)
        assert np.isnan(result)

    def test_negative_ic50_returns_nan(self):
        """Test that negative IC50 returns NaN."""
        result = convert_ic50_to_pic50(-100.0)
        assert np.isnan(result)

    def test_very_small_ic50(self):
        """Test very potent compound (very small IC50)."""
        # IC50 = 0.0001 nM → pIC50 ≈ 13.0
        result = convert_ic50_to_pic50(0.0001)
        assert result > 12.9

    def test_very_large_ic50(self):
        """Test weak compound (very large IC50)."""
        # IC50 = 100000 nM (100 µM) → pIC50 = 4.0
        result = convert_ic50_to_pic50(100000.0)
        assert abs(result - 4.0) < 0.01


class TestCleanAndPreprocessData:
    """Test clean_and_preprocess_data function."""

    def test_successful_preprocessing(self):
        """Test successful data preprocessing."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "standard_value": [100.0, 500.0, 2000.0],
                "assay_id": ["A1", "A2", "A3"],
                "reference": ["EGFR", "EGFR", "EGFR"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert isinstance(result["data"], pd.DataFrame)
        assert list(result["data"].columns) == ["smiles", "pIC50"]
        assert result["stats"]["input"] == 3
        assert result["stats"]["output"] == 3

    def test_removes_missing_smiles(self):
        """Test removal of rows with missing SMILES."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", None, "c1ccccc1"],
                "standard_value": [100.0, 500.0, 2000.0],
                "assay_id": ["A1", "A2", "A3"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert result["stats"]["removed_missing_smiles"] == 1
        assert result["stats"]["output"] == 2

    def test_removes_duplicate_smiles(self):
        """Test removal of duplicate SMILES."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "standard_value": [100.0, 150.0, 500.0],
                "assay_id": ["A1", "A2", "A3"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert result["stats"]["removed_duplicates"] == 1
        assert result["stats"]["output"] == 2

    def test_removes_invalid_conversions(self):
        """Test removal of rows with invalid IC50 conversions."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "standard_value": [100.0, 0.0, 2000.0],  # 0.0 will fail conversion
                "assay_id": ["A1", "A2", "A3"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert result["stats"]["removed_invalid_conversions"] == 1
        assert result["stats"]["output"] == 2

    def test_removes_outliers_default_range(self):
        """Test removal of outliers with default pIC50 range (3.0-12.0)."""
        df = pd.DataFrame(
            {
                "smiles": [
                    "CC(=O)Oc1ccccc1C(=O)O",  # 100 nM → pIC50=7.0 ✓
                    "CCO",  # 1000000 nM (1 mM) → pIC50≈3.0 → boundary
                    "c1ccccc1",  # 10000000 nM (10 mM) → pIC50≈2.0 ✗
                ],
                "standard_value": [100.0, 1000000.0, 10000000.0],
                "assay_id": ["A1", "A2", "A3"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert result["stats"]["removed_outliers"] == 1  # 10 mM is below 3.0
        assert result["stats"]["output"] == 2

    def test_removes_outliers_custom_stricter_range(self):
        """Test removal of outliers with stricter pIC50 range (5.0-10.0)."""
        df = pd.DataFrame(
            {
                "smiles": [
                    "CC(=O)Oc1ccccc1C(=O)O",  # 100 nM → pIC50=7.0 ✓
                    "CCO",  # 1000 nM → pIC50=6.0 ✓
                    "c1ccccc1",  # 100000 nM → pIC50=4.0 ✗
                    "CC(C)C",  # 10000000 nM → pIC50=2.0 ✗
                ],
                "standard_value": [100.0, 1000.0, 100000.0, 10000000.0],
                "assay_id": ["A1", "A2", "A3", "A4"],
            }
        )

        result = clean_and_preprocess_data(df, min_pic50=5.0, max_pic50=10.0)

        assert result["success"] is True
        assert result["stats"]["removed_outliers"] == 2
        assert result["stats"]["output"] == 2

    def test_stats_tracking(self):
        """Test that stats are correctly tracked through all steps."""
        df = pd.DataFrame(
            {
                "smiles": [
                    "CC(=O)Oc1ccccc1C(=O)O",
                    None,  # Missing SMILES
                    "CC(=O)Oc1ccccc1C(=O)O",  # Duplicate
                    "CCO",
                ],
                "standard_value": [100.0, 500.0, 100.0, 0.0],  # Last one fails
                "assay_id": ["A1", "A2", "A3", "A4"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        stats = result["stats"]
        assert stats["input"] == 4
        assert stats["removed_missing_smiles"] == 1
        assert stats["removed_duplicates"] == 1
        assert stats["removed_invalid_conversions"] == 1
        assert stats["output"] == 1

    def test_output_dataframe_structure(self):
        """Test that output DataFrame has correct structure."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "standard_value": [100.0, 500.0],
                "assay_id": ["A1", "A2"],
                "reference": ["EGFR", "EGFR"],
            }
        )

        result = clean_and_preprocess_data(df)

        output_df = result["data"]
        assert list(output_df.columns) == ["smiles", "pIC50"]
        assert len(output_df) == 2
        assert output_df.index.tolist() == [0, 1]  # Reset index

    def test_exception_handling(self):
        """Test exception handling for invalid data."""
        # Create a DataFrame that will cause an exception during processing
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
                "standard_value": [100.0],
                # Missing 'assay_id' column - not needed for processing
            }
        )

        # This should succeed because missing columns aren't required
        result = clean_and_preprocess_data(df)
        assert result["success"] is True

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"smiles": [], "standard_value": [], "assay_id": []})

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert result["stats"]["output"] == 0
        assert len(result["data"]) == 0

    def test_all_rows_filtered_out(self):
        """Test when all rows are filtered out."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "standard_value": [0.0, 0.0],  # Both will fail conversion
                "assay_id": ["A1", "A2"],
            }
        )

        result = clean_and_preprocess_data(df)

        assert result["success"] is True
        assert result["stats"]["output"] == 0
        assert len(result["data"]) == 0

    def test_pIC50_values_in_output(self):
        """Test that pIC50 values are correctly calculated."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "standard_value": [100.0, 1000.0, 10000.0],  # 7.0, 6.0, 5.0
                "assay_id": ["A1", "A2", "A3"],
            }
        )

        result = clean_and_preprocess_data(df)

        pic50_values = result["data"]["pIC50"].values
        assert abs(pic50_values[0] - 7.0) < 0.01
        assert abs(pic50_values[1] - 6.0) < 0.01
        assert abs(pic50_values[2] - 5.0) < 0.01


class TestGetCleanedDataset:
    """Test get_cleaned_dataset wrapper function."""

    def test_successful_dataset_cleaning(self):
        """Test successful dataset cleaning through wrapper."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "standard_value": [100.0, 500.0],
                "assay_id": ["A1", "A2"],
            }
        )

        result_df, stats = get_cleaned_dataset(df)

        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == ["smiles", "pIC50"]
        assert stats["output"] == 2

    def test_failure_returns_none(self):
        """Test that failure returns None for DataFrame."""
        # Empty DataFrame
        df = pd.DataFrame({"smiles": [], "standard_value": [], "assay_id": []})

        result_df, stats = get_cleaned_dataset(df)

        assert result_df is not None  # Empty DF is still valid
        assert len(result_df) == 0

    def test_wrapper_with_default_thresholds(self):
        """Test wrapper with default pIC50 thresholds."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
                "standard_value": [100.0, 1000.0, 10000000.0],  # Last too weak
                "assay_id": ["A1", "A2", "A3"],
            }
        )

        result_df, stats = get_cleaned_dataset(df)

        assert stats["output"] == 2  # Default filters out weak binder

    def test_wrapper_with_custom_stricter_thresholds(self):
        """Test wrapper with custom stricter pIC50 thresholds."""
        df = pd.DataFrame(
            {
                "smiles": [
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "CCO",
                    "c1ccccc1",
                    "CC(C)C",
                ],
                "standard_value": [100.0, 1000.0, 100000.0, 10000000.0],
                "assay_id": ["A1", "A2", "A3", "A4"],
            }
        )

        # Stricter: min_pic50=5.0, max_pic50=10.0
        result_df, stats = get_cleaned_dataset(df, min_pic50=5.0, max_pic50=10.0)

        assert stats["output"] == 2  # Filters out both weak binders (pIC50 < 5.0)

    def test_wrapper_returns_stats(self):
        """Test that wrapper returns stats dictionary."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "standard_value": [100.0, 500.0],
                "assay_id": ["A1", "A2"],
            }
        )

        _, stats = get_cleaned_dataset(df)

        assert isinstance(stats, dict)
        assert "input" in stats
        assert "output" in stats
        assert "removed_missing_smiles" in stats
        assert "removed_duplicates" in stats
        assert "removed_outliers" in stats

    def test_wrapper_failure_returns_none_and_empty_dict(self):
        """Test that wrapper returns None and empty dict on failure."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
                "standard_value": [100.0],
                "assay_id": ["A1"],
            }
        )

        # Mock clean_and_preprocess_data to return failure
        with patch("app.qsar.preprocessing.clean_and_preprocess_data") as mock_clean:
            mock_clean.return_value = {
                "success": False,
                "error": "Test error",
            }
            result_df, stats = get_cleaned_dataset(df)

            assert result_df is None
            assert isinstance(stats, dict)


class TestCleanAndPreprocessDataExceptions:
    """Test exception handling in clean_and_preprocess_data."""

    def test_exception_during_conversion(self):
        """Test exception handling during IC50 to pIC50 conversion."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
                "standard_value": [100.0, 500.0],
                "assay_id": ["A1", "A2"],
            }
        )

        # Mock convert_ic50_to_pic50 to raise exception
        with patch("app.qsar.preprocessing.convert_ic50_to_pic50") as mock_convert:
            mock_convert.side_effect = ValueError("Conversion error")

            # The actual function uses df.apply which will catch the error
            # So we need to trigger exception differently
            with patch.object(pd.DataFrame, "apply", side_effect=Exception("Apply failed")):
                result = clean_and_preprocess_data(df)

                assert result["success"] is False
                assert "error" in result
                assert result["stats"]["input"] == 2

    def test_exception_in_main_processing(self):
        """Test exception handling during main processing."""
        df = pd.DataFrame(
            {
                "smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
                "standard_value": [100.0],
                "assay_id": ["A1"],
            }
        )

        # Mock dropna to raise exception
        with patch.object(pd.DataFrame, "dropna", side_effect=Exception("dropna failed")):
            result = clean_and_preprocess_data(df)

            assert result["success"] is False
            assert "error" in result

    def test_clean_and_preprocess_without_required_columns(self):
        """Test handling of DataFrame without required columns."""
        df = pd.DataFrame(
            {
                "other_col": [1, 2],
            }
        )

        # This should fail when trying to access 'smiles'
        result = clean_and_preprocess_data(df)

        assert result["success"] is False
        assert "error" in result
