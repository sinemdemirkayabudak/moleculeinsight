"""Tests for QSAR prediction pipeline core functionality.

Tests basic pipeline initialization, data loading, preprocessing,
featurization, training, predictions, and explanations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from app.qsar.qsar_prediction import QSARPipeline


class TestQSARPipelineInit:
    """Test QSARPipeline initialization."""

    def test_init_creates_instance(self):
        """Test that QSARPipeline initializes correctly."""
        pipeline = QSARPipeline()

        assert pipeline is not None
        assert pipeline.raw_data is None
        assert pipeline.cleaned_data is None
        assert pipeline.features is None
        assert pipeline.feature_names is None
        assert pipeline.feature_type is None
        assert pipeline.rf_model is None
        assert pipeline.xgb_model is None
        assert pipeline.splits is None
        assert pipeline.metrics is None

    def test_init_creates_components(self):
        """Test that pipeline initializes all components."""
        pipeline = QSARPipeline()

        assert pipeline.trainer is not None
        assert pipeline.predictor is not None
        assert pipeline.explainer is not None


class TestLoadData:
    """Test data loading functionality."""

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    def test_load_data_success(self, mock_get_data):
        """Test successful data loading."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC", "CCC"],
                "value": [5.0, 6.0],
            }
        )

        mock_get_data.return_value = {
            "success": True,
            "data": mock_data,
        }

        pipeline = QSARPipeline()
        result = pipeline.load_data(limit=100, offset=0)

        assert result["success"] is True
        assert result["count"] == 2
        assert pipeline.raw_data is not None
        assert len(pipeline.raw_data) == 2

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    def test_load_data_failure(self, mock_get_data):
        """Test data loading failure."""
        mock_get_data.return_value = {
            "success": False,
            "error": "API error",
        }

        pipeline = QSARPipeline()
        result = pipeline.load_data()

        assert result["success"] is False
        assert "error" in result
        assert pipeline.raw_data is None

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    def test_load_data_with_parameters(self, mock_get_data):
        """Test data loading with custom parameters."""
        mock_data = pd.DataFrame({"canonical_smiles": ["C"], "value": [5.0]})
        mock_get_data.return_value = {"success": True, "data": mock_data}

        pipeline = QSARPipeline()
        pipeline.load_data(limit=500, offset=100)

        mock_get_data.assert_called_once_with(limit=500, offset=100)


class TestPreprocessData:
    """Test data preprocessing."""

    def test_preprocess_data_success(self):
        """Test successful preprocessing."""
        pipeline = QSARPipeline()
        pipeline.raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC", "CCC", "CCCC"],
                "value": [5.0, 6.0, 7.0],
            }
        )

        with patch("app.qsar.qsar_prediction.get_cleaned_dataset") as mock_clean:
            cleaned = pd.DataFrame(
                {
                    "smiles": ["CC", "CCC"],
                    "pIC50": [5.0, 6.0],
                }
            )
            mock_clean.return_value = (cleaned, {"removed": 1})

            result = pipeline.preprocess_data()

            assert result["success"] is True
            assert pipeline.cleaned_data is not None
            assert len(pipeline.cleaned_data) == 2

    def test_preprocess_data_no_raw_data(self):
        """Test preprocessing with no raw data."""
        pipeline = QSARPipeline()
        result = pipeline.preprocess_data()

        assert result["success"] is False
        assert "error" in result

    def test_preprocess_data_failure(self):
        """Test preprocessing failure."""
        pipeline = QSARPipeline()
        pipeline.raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC"],
                "value": [5.0],
            }
        )

        with patch("app.qsar.qsar_prediction.get_cleaned_dataset") as mock_clean:
            mock_clean.return_value = (None, {"error": "Failed"})

            result = pipeline.preprocess_data()

            assert result["success"] is False

    def test_preprocess_data_with_parameters(self):
        """Test preprocessing with custom parameters."""
        pipeline = QSARPipeline()
        pipeline.raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC"],
                "value": [5.0],
            }
        )

        with patch("app.qsar.qsar_prediction.get_cleaned_dataset") as mock_clean:
            cleaned = pd.DataFrame({"smiles": ["CC"], "pIC50": [5.0]})
            mock_clean.return_value = (cleaned, {})

            pipeline.preprocess_data(min_pic50=2.0, max_pic50=10.0)

            mock_clean.assert_called_once_with(
                pipeline.raw_data,
                min_pic50=2.0,
                max_pic50=10.0,
            )


class TestFeaturizeData:
    """Test feature computation."""

    def test_featurize_data_morgan_success(self):
        """Test successful Morgan fingerprint featurization."""
        pipeline = QSARPipeline()
        pipeline.cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC", "CCC"],
                "pIC50": [5.0, 6.0],
            }
        )

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            X = np.random.randn(2, 2048)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": [f"f_{i}" for i in range(2048)],
            }

            result = pipeline.featurize_data(fingerprint_type="morgan", radius=2)

            assert result["success"] is True
            assert pipeline.features is not None
            assert pipeline.feature_type == "morgan"
            assert len(pipeline.feature_names) == 2048  # ty:ignore[invalid-argument-type]

    def test_featurize_data_rdkit_success(self):
        """Test successful RDKit descriptor featurization."""
        pipeline = QSARPipeline()
        pipeline.cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC", "CCC"],
                "pIC50": [5.0, 6.0],
            }
        )

        with patch("app.qsar.qsar_prediction.compute_rdkit_descriptors") as mock_feat:
            X = np.random.randn(2, 200)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": [f"f_{i}" for i in range(200)],
            }

            result = pipeline.featurize_data(fingerprint_type="rdkit")

            assert result["success"] is True
            assert pipeline.feature_type == "rdkit"

    def test_featurize_data_no_cleaned_data(self):
        """Test featurization with no cleaned data."""
        pipeline = QSARPipeline()
        result = pipeline.featurize_data()

        assert result["success"] is False
        assert "error" in result

    def test_featurize_data_unsupported_type(self):
        """Test featurization with unsupported feature type."""
        pipeline = QSARPipeline()
        pipeline.cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC"],
                "pIC50": [5.0],
            }
        )

        result = pipeline.featurize_data(fingerprint_type="unsupported")

        assert result["success"] is False
        assert "Unsupported" in result["error"]

    def test_featurize_data_morgan_failure(self):
        """Test Morgan featurization failure."""
        pipeline = QSARPipeline()
        pipeline.cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC"],
                "pIC50": [5.0],
            }
        )

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            mock_feat.return_value = {
                "success": False,
                "error": "Feature error",
            }

            result = pipeline.featurize_data()

            assert result["success"] is False

    def test_featurize_data_none_feature_names(self):
        """Test featurization generates feature names when None."""
        pipeline = QSARPipeline()
        pipeline.cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC"],
                "pIC50": [5.0],
            }
        )

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            X = np.random.randn(1, 2048)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": None,
            }

            result = pipeline.featurize_data()

            assert result["success"] is True


class TestTrainModels:
    """Test model training."""

    def test_train_models_success(self):
        """Test successful model training."""
        pipeline = QSARPipeline()
        pipeline.cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC"] * 20,
                "pIC50": np.random.randn(20),
            }
        )
        pipeline.features = np.random.randn(20, 10)

        with patch.object(pipeline.trainer, "train_both_models") as mock_train:
            rf_model = MagicMock()
            xgb_model = MagicMock()
            mock_train.return_value = (
                rf_model,
                xgb_model,
                {
                    "splits": {
                        "X_train": np.random.randn(15, 10),
                        "X_test": np.random.randn(5, 10),
                        "y_train": np.random.randn(15),
                        "y_test": np.random.randn(5),
                    },
                    "metrics": {"rf_r2": 0.8, "xgb_r2": 0.85},
                },
            )

            result = pipeline.train_models()

            assert result["success"] is True
            assert pipeline.rf_model is not None
            assert pipeline.xgb_model is not None
            assert pipeline.splits is not None

    def test_train_models_no_features(self):
        """Test training with no features."""
        pipeline = QSARPipeline()
        result = pipeline.train_models()

        assert result["success"] is False

    def test_train_models_no_cleaned_data(self):
        """Test training with no cleaned data."""
        pipeline = QSARPipeline()
        pipeline.features = np.random.randn(10, 10)
        result = pipeline.train_models()

        assert result["success"] is False


class TestGetPredictions:
    """Test prediction generation."""

    def test_get_predictions_rf_success(self):
        """Test successful RF predictions."""
        pipeline = QSARPipeline()
        pipeline.splits = {
            "X_test": np.random.randn(10, 10),
            "y_test": np.random.randn(10),
        }
        pipeline.rf_model = MagicMock()

        with patch.object(pipeline.predictor, "compute_confidence_intervals") as mock_pred:
            mock_pred.return_value = pd.DataFrame(
                {
                    "pIC50": [5.0] * 10,
                    "ci_lower": [4.5] * 10,
                    "ci_upper": [5.5] * 10,
                }
            )

            result = pipeline.get_predictions(model_type="rf")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 10

    def test_get_predictions_xgb_success(self):
        """Test successful XGBoost predictions."""
        pipeline = QSARPipeline()
        pipeline.splits = {
            "X_test": np.random.randn(10, 10),
            "y_test": np.random.randn(10),
        }
        pipeline.xgb_model = MagicMock()

        with patch.object(pipeline.predictor, "compute_confidence_intervals") as mock_pred:
            mock_pred.return_value = pd.DataFrame(
                {
                    "pIC50": [5.0] * 10,
                }
            )

            result = pipeline.get_predictions(model_type="xgb")

            assert isinstance(result, pd.DataFrame)

    def test_get_predictions_no_splits(self):
        """Test predictions with no splits."""
        pipeline = QSARPipeline()
        result = pipeline.get_predictions()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_predictions_model_not_trained(self):
        """Test predictions when model not trained."""
        pipeline = QSARPipeline()
        pipeline.splits = {"X_test": np.random.randn(10, 10)}
        pipeline.rf_model = None

        result = pipeline.get_predictions(model_type="rf")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestGetExplanations:
    """Test explanation generation."""

    def test_get_explanations_rf_success(self):
        """Test successful RF explanations."""
        pipeline = QSARPipeline()
        pipeline.splits = {
            "X_test": np.random.randn(10, 10),
            "y_test": np.random.randn(10),
        }
        pipeline.rf_model = MagicMock()
        pipeline.feature_names = [f"f_{i}" for i in range(10)]

        with patch.object(pipeline.explainer, "create_explainer") as mock_create:
            with patch.object(pipeline.explainer, "compute_shap_values") as mock_shap:
                with patch.object(pipeline.explainer, "get_feature_importance") as mock_importance:
                    with patch.object(pipeline.explainer, "summary_stats") as mock_stats:
                        shap_vals = np.random.randn(10, 10)
                        mock_create.return_value = MagicMock()
                        mock_shap.return_value = shap_vals
                        mock_importance.return_value = [("f_0", 0.5)]
                        mock_stats.return_value = {"mean": 5.0}

                        result = pipeline.get_explanations(model_type="rf")

                        assert "importance" in result
                        assert "stats" in result
                        assert "shap_values" in result

    def test_get_explanations_xgb_success(self):
        """Test successful XGBoost explanations."""
        pipeline = QSARPipeline()
        pipeline.splits = {
            "X_test": np.random.randn(10, 10),
            "y_test": np.random.randn(10),
        }
        pipeline.xgb_model = MagicMock()
        pipeline.feature_names = [f"f_{i}" for i in range(10)]

        with patch.object(pipeline.explainer, "create_explainer"):
            with patch.object(pipeline.explainer, "compute_shap_values"):
                with patch.object(pipeline.explainer, "get_feature_importance"):
                    with patch.object(pipeline.explainer, "summary_stats"):
                        result = pipeline.get_explanations(model_type="xgb", top_n=10)

                        assert isinstance(result, dict)

    def test_get_explanations_no_splits(self):
        """Test explanations with no splits."""
        pipeline = QSARPipeline()
        result = pipeline.get_explanations()

        assert result == {}

    def test_get_explanations_model_not_trained(self):
        """Test explanations when model not trained."""
        pipeline = QSARPipeline()
        pipeline.splits = {"X_test": np.random.randn(10, 10)}
        pipeline.rf_model = None

        result = pipeline.get_explanations(model_type="rf")

        assert result == {}
