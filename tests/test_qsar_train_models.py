"""Comprehensive tests for app.qsar.train_models module (100% coverage).

Tests cover:
- Data loading with pagination and batch handling
- Data preprocessing and cleaning
- Feature computation (Morgan fingerprints + RDKit descriptors)
- Feature combination/concatenation
- Model training pipeline
- Model evaluation on test set
- Cross-validation metrics
- File I/O (model saving, metadata, performance metrics)
- Error handling for failed operations
- Edge cases (empty batches, missing data, failed preprocessing)
"""

# Suppress streamlit warnings during test imports
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from app.qsar.train_models import main

warnings.filterwarnings("ignore")


class TestDataLoading:
    """Test data loading and batch processing."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_successful_data_loading_multiple_batches(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test loading multiple batches of data from API."""
        # Setup mocks
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        # Create sample data for batches
        sample_data = pd.DataFrame(
            {"smiles": ["CCO", "CC(C)C"], "standard_value": [5.0, 6.0], "pIC50": [5.0, 6.0]}
        )

        # Mock successful batch loading - 2 successes then all failures
        call_count = [0]

        def load_data_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"success": True, "data": sample_data}
            else:
                return {"success": False}

        mock_pipeline.load_data.side_effect = load_data_side_effect

        # Mock preprocessing
        mock_preprocess = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = mock_preprocess

        # Mock training
        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1, 2], [3, 4]]),
                    "X_test": np.array([[5, 6]]),
                    "y_train": np.array([1, 2]),
                    "y_test": np.array([3]),
                },
                "metrics": {
                    "rf": {"cv_r2_mean": 0.5, "cv_r2_std": 0.1},
                    "xgb": {"cv_r2_mean": 0.7, "cv_r2_std": 0.05},
                },
            },
        )

        # Mock evaluation
        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.7, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.6
        mock_xgb.score.return_value = 0.72

        # Mock feature computation
        mock_morgan.return_value = {"success": True, "X": np.array([[1, 0], [0, 1]])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[2], [3]])}

        # Mock file operations
        with (
            patch("pathlib.Path.mkdir"),
            patch("joblib.dump"),
            patch("builtins.open", create=True) as mock_file,
        ):
            mock_file.return_value.__enter__.return_value.write = MagicMock()

            # Run main
            main()

        # Verify batching - main loops n_batches=10 times, but this test only does 2 successful loads
        # The side_effect is checked across all 10 batch attempts
        assert mock_pipeline.load_data.call_count >= 2

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_no_data_loaded_failure(self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class):
        """Test handling when no data is successfully loaded."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        # All batches fail
        mock_pipeline.load_data.return_value = {"success": False}

        # Should return early without crashing
        main()

        # Should print error message
        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("Failed to load any data" in str(call) for call in print_calls)

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_empty_batch_stops_loading(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test that empty batch stops further loading attempts."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        # First batch succeeds, second returns None data (stops loading)
        mock_pipeline.load_data.side_effect = [
            {"success": True, "data": sample_data},
            {"success": True, "data": None},
        ]

        # Mock preprocessing
        mock_preprocess = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = mock_preprocess

        # Mock training
        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1]]),
                    "X_test": np.array([[2]]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {},
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.5, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.5

        mock_morgan.return_value = {"success": True, "X": np.array([[1]])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[2]])}

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        # Should have called load_data exactly 2 times
        assert mock_pipeline.load_data.call_count == 2


class TestDataPreprocessing:
    """Test data preprocessing pipeline."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_preprocessing_success(self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class):
        """Test successful data preprocessing."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        raw_data = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(C)C"],
                "standard_value": ["5.0", "6.0"],  # String format (needs conversion)
                "pIC50": [5.0, 6.0],
            }
        )

        cleaned_data = pd.DataFrame(
            {"smiles": ["CCO", "CC(C)C"], "standard_value": [5.0, 6.0], "pIC50": [5.0, 6.0]}
        )

        mock_pipeline.load_data.return_value = {"success": True, "data": raw_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": cleaned_data}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1, 2]]),
                    "X_test": np.array([[3, 4]]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {},
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.6

        mock_morgan.return_value = {"success": True, "X": np.array([[1]])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[2]])}

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        mock_pipeline.preprocess_data.assert_called_once()

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_preprocessing_failure(self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class):
        """Test handling of preprocessing failure."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})
        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}

        # Preprocessing fails
        mock_pipeline.preprocess_data.return_value = {"success": False, "error": "Invalid data"}

        # Should return early
        main()

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("Preprocessing failed" in str(call) for call in print_calls)


class TestFeatureComputation:
    """Test Morgan fingerprints and RDKit descriptor computation."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_morgan_computation_success(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test successful Morgan fingerprint computation."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        # Morgan succeeds with 2048 bits
        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}

        # RDKit succeeds with 8 descriptors
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8])}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {},
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.6

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        mock_morgan.assert_called_once()
        mock_rdkit.assert_called_once()

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_morgan_computation_failure(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test handling of Morgan fingerprint computation failure."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame(
            {"smiles": ["invalid_smiles"], "standard_value": [5.0], "pIC50": [5.0]}
        )

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        # Morgan fails
        mock_morgan.return_value = {"success": False, "error": "Invalid SMILES"}

        main()

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("Failed" in str(call) for call in print_calls)

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_rdkit_computation_failure(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test handling of RDKit descriptor computation failure."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}

        # RDKit fails
        mock_rdkit.return_value = {"success": False, "error": "Descriptor computation failed"}

        main()

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("Failed" in str(call) for call in print_calls)


class TestFeatureCombination:
    """Test feature concatenation and combination."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_feature_combination_shapes(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test that combined features have correct shape (2048 + 8 = 2056)."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame(
            {"smiles": ["CCO", "CC(C)C"], "standard_value": [5.0, 6.0], "pIC50": [5.0, 6.0]}
        )

        # Mock load_data with side effect function to handle multiple calls
        call_count = [0]

        def load_data_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"success": True, "data": sample_data}
            else:
                return {"success": False}

        mock_pipeline.load_data.side_effect = load_data_side_effect
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        # Create feature arrays with exact shapes
        morgan_features = np.random.rand(2, 2048)
        rdkit_features = np.random.rand(2, 8)

        mock_morgan.return_value = {"success": True, "X": morgan_features}
        mock_rdkit.return_value = {"success": True, "X": rdkit_features}

        # Capture the training call to verify shape
        captured_X = None

        def capture_X(X, y, cross_val=False):
            nonlocal captured_X
            captured_X = X
            mock_rf = MagicMock()
            mock_rf.score.return_value = 0.5  # Ensure this is a float
            mock_xgb = MagicMock()
            mock_xgb.score.return_value = 0.6  # Ensure this is a float
            return (
                mock_rf,
                mock_xgb,
                {
                    "splits": {
                        "X_train": X[:1],
                        "X_test": X[1:],
                        "y_train": y[:1],
                        "y_test": y[1:],
                    },
                    "metrics": {
                        "rf": {"cv_r2_mean": 0.5, "cv_r2_std": 0.1},
                        "xgb": {"cv_r2_mean": 0.6, "cv_r2_std": 0.05},
                    },
                },
            )

        mock_pipeline.trainer.train_both_models.side_effect = capture_X
        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        # Verify combined shape is 2048 + 8 = 2056
        assert captured_X is not None
        assert captured_X.shape == (2, 2056)


class TestModelTraining:
    """Test model training and evaluation."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_both_models_trained(self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class):
        """Test that both RF and XGB models are trained."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8])}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {
                    "rf": {"cv_r2_mean": 0.5, "cv_r2_std": 0.1},
                    "xgb": {"cv_r2_mean": 0.7, "cv_r2_std": 0.05},
                },
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.7

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        mock_pipeline.trainer.train_both_models.assert_called_once()
        assert mock_pipeline.trainer.evaluate_model.call_count == 2  # Both RF and XGB evaluated


class TestModelEvaluation:
    """Test model evaluation metrics."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_cross_validation_metrics_included(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test that CV metrics are included in final results."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8])}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {
                    "rf": {"cv_r2_mean": 0.55, "cv_r2_std": 0.08},
                    "xgb": {"cv_r2_mean": 0.71, "cv_r2_std": 0.04},
                },
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.7

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        # Verify CV metrics are mentioned in print calls
        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        cv_mentioned = any("5-Fold CV R²" in str(call) for call in print_calls)
        assert cv_mentioned


class TestFileSaving:
    """Test model and metadata file saving."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("pathlib.Path.mkdir")
    @patch("joblib.dump")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    def test_models_saved_to_disk(
        self,
        mock_print,
        mock_file,
        mock_dump,
        mock_mkdir,
        mock_rdkit,
        mock_morgan,
        mock_pipeline_class,
    ):
        """Test that trained models are saved to disk."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8])}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {},
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.7

        mock_file.return_value.__enter__.return_value.write = MagicMock()

        main()

        # Verify joblib.dump called for models
        assert mock_dump.call_count >= 2  # RF and XGB models

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("pathlib.Path.mkdir")
    @patch("joblib.dump")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    def test_metadata_json_saved(
        self,
        mock_print,
        mock_file,
        mock_dump,
        mock_mkdir,
        mock_rdkit,
        mock_morgan,
        mock_pipeline_class,
    ):
        """Test that metadata JSON is saved with correct structure."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame(
            {"smiles": ["CCO", "CC(C)C"], "standard_value": [5.0, 6.0], "pIC50": [5.0, 6.0]}
        )

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048, [0] * 2048])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8, [0] * 8])}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056, [0] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([1, 2]),
                    "y_test": np.array([3]),
                },
                "metrics": {},
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.6, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.7

        # Capture JSON writes
        written_json = {}

        def mock_open_func(*args, **kwargs):
            m = MagicMock()
            m.__enter__.return_value.write = lambda x: written_json.update({str(args[0]): x})
            return m

        with patch("builtins.open", side_effect=mock_open_func):
            main()

        # Verify JSON files were written
        metadata_written = any("metadata" in k for k in written_json.keys())
        assert metadata_written

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_performance_metrics_file_saved(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test that performance metrics file is saved."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        mock_pipeline.load_data.return_value = {"success": True, "data": sample_data}
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8])}

        mock_rf = MagicMock()
        mock_xgb = MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([1]),
                    "y_test": np.array([2]),
                },
                "metrics": {},
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.7, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.5
        mock_xgb.score.return_value = 0.7

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        perf_saved = any("performance" in str(call) for call in print_calls)
        assert perf_saved


class TestCompleteWorkflow:
    """Integration tests for complete training workflow."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_complete_training_workflow(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test complete end-to-end training workflow."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        # Realistic sample data
        sample_data = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(C)C", "c1ccccc1", "CC(=O)O"],
                "standard_value": [5.0, 6.0, 7.0, 5.5],
                "pIC50": [5.0, 6.0, 7.0, 5.5],
            }
        )

        mock_pipeline.load_data.side_effect = [
            {"success": True, "data": sample_data},
            {"success": False},
            {"success": False},
            {"success": False},
            {"success": False},
            {"success": False},
            {"success": False},
            {"success": False},
            {"success": False},
            {"success": False},
        ]

        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        # Realistic feature shapes
        n_samples = 4
        morgan_features = np.random.rand(n_samples, 2048)
        rdkit_features = np.random.rand(n_samples, 8)

        mock_morgan.return_value = {"success": True, "X": morgan_features}
        mock_rdkit.return_value = {"success": True, "X": rdkit_features}

        # Mock training
        mock_rf = MagicMock()
        mock_xgb = MagicMock()

        X_combined = np.hstack([morgan_features, rdkit_features])
        y = np.array(sample_data["pIC50"])

        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": X_combined[:3],
                    "X_test": X_combined[3:],
                    "y_train": y[:3],
                    "y_test": y[3:],
                },
                "metrics": {
                    "rf": {"cv_r2_mean": 0.55, "cv_r2_std": 0.05},
                    "xgb": {"cv_r2_mean": 0.70, "cv_r2_std": 0.03},
                },
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.68, "rmse": 0.55, "mae": 0.3}
        mock_rf.score.return_value = 0.60
        mock_xgb.score.return_value = 0.70

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        # Verify complete workflow executed
        assert mock_pipeline.load_data.called
        assert mock_pipeline.preprocess_data.called
        assert mock_morgan.called
        assert mock_rdkit.called
        assert mock_pipeline.trainer.train_both_models.called

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("TRAINING COMPLETE" in str(call) for call in print_calls)


class TestErrorHandlingPaths:
    """Test error handling in expanded training pipeline."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_batch_loading_with_mixed_success(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test loading when some batches fail and some succeed."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        # Create side effect that returns success for first 2 calls, then failure
        load_results = [
            {"success": True, "data": sample_data},
            {"success": False},
            {"success": False},
        ] + [{"success": False}] * 7  # Rest of 10 batches fail
        mock_pipeline.load_data.side_effect = load_results

        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}
        mock_morgan.return_value = {"success": True, "X": np.array([[1] * 2048])}
        mock_rdkit.return_value = {"success": True, "X": np.array([[1] * 8])}

        mock_rf, mock_xgb = MagicMock(), MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.array([[1] * 2056]),
                    "X_test": np.array([[1] * 2056]),
                    "y_train": np.array([5.0]),
                    "y_test": np.array([5.0]),
                },
                "metrics": {},
            },
        )
        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.70, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.70
        mock_xgb.score.return_value = 0.70

        with patch("pathlib.Path.mkdir"), patch("joblib.dump"), patch("builtins.open", create=True):
            main()

        # Should handle mixed success/failure gracefully
        assert mock_pipeline.load_data.call_count >= 2

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_morgan_fingerprint_failure(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test handling when Morgan fingerprint computation fails."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        call_count = [0]

        def load_data_side_effect(**kwargs):
            call_count[0] += 1
            return (
                {"success": True, "data": sample_data} if call_count[0] == 1 else {"success": False}
            )

        mock_pipeline.load_data.side_effect = load_data_side_effect
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}

        # Morgan fails
        mock_morgan.return_value = {"success": False, "error": "Invalid SMILES"}

        main()

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("Failed" in str(call) for call in print_calls)

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("builtins.print")
    def test_preprocessing_error_handling(
        self, mock_print, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test preprocessing error message is printed correctly."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame({"smiles": ["CCO"], "standard_value": [5.0], "pIC50": [5.0]})

        call_count = [0]

        def load_data_side_effect(**kwargs):
            call_count[0] += 1
            return (
                {"success": True, "data": sample_data} if call_count[0] == 1 else {"success": False}
            )

        mock_pipeline.load_data.side_effect = load_data_side_effect
        mock_pipeline.preprocess_data.return_value = {
            "success": False,
            "error": "Missing required columns",
        }

        main()

        print_calls = [str(call[0]) for call in mock_print.call_args_list]
        assert any("Preprocessing failed" in str(call) for call in print_calls)


class TestMetadataAndPerformanceSaving:
    """Test metadata and performance metrics file generation."""

    @patch("app.qsar.train_models.QSARPipeline")
    @patch("app.qsar.train_models.compute_morgan_fingerprints")
    @patch("app.qsar.train_models.compute_rdkit_descriptors")
    @patch("app.qsar.train_models.Path")
    @patch("builtins.print")
    def test_metadata_json_structure(
        self, mock_print, mock_path, mock_rdkit, mock_morgan, mock_pipeline_class
    ):
        """Test that metadata JSON has correct structure."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        sample_data = pd.DataFrame(
            {"smiles": ["CCO", "CC(C)C"], "standard_value": [5.0, 6.0], "pIC50": [5.0, 6.0]}
        )

        call_count = [0]

        def load_data_side_effect(**kwargs):
            call_count[0] += 1
            return (
                {"success": True, "data": sample_data} if call_count[0] == 1 else {"success": False}
            )

        mock_pipeline.load_data.side_effect = load_data_side_effect
        mock_pipeline.preprocess_data.return_value = {"success": True, "data": sample_data}
        mock_morgan.return_value = {"success": True, "X": np.random.rand(2, 2048)}
        mock_rdkit.return_value = {"success": True, "X": np.random.rand(2, 8)}

        mock_rf, mock_xgb = MagicMock(), MagicMock()
        mock_pipeline.trainer.train_both_models.return_value = (
            mock_rf,
            mock_xgb,
            {
                "splits": {
                    "X_train": np.random.rand(1, 2056),
                    "X_test": np.random.rand(1, 2056),
                    "y_train": np.array([5.0]),
                    "y_test": np.array([6.0]),
                },
                "metrics": {
                    "rf": {"cv_r2_mean": 0.6, "cv_r2_std": 0.05},
                    "xgb": {"cv_r2_mean": 0.7, "cv_r2_std": 0.03},
                },
            },
        )

        mock_pipeline.trainer.evaluate_model.return_value = {"r2": 0.70, "rmse": 0.5, "mae": 0.3}
        mock_rf.score.return_value = 0.60
        mock_xgb.score.return_value = 0.70

        mock_path_inst = MagicMock()
        mock_path.return_value = mock_path_inst
        mock_path_inst.__truediv__.return_value = mock_path_inst

        with patch("joblib.dump"), patch("builtins.open", create=True) as mock_file:
            main()

        # Verify files were written
        assert mock_file.call_count >= 2  # metadata.json and performance.json
        assert mock_pipeline.trainer.train_both_models.called
        assert mock_pipeline.trainer.evaluate_model.called
