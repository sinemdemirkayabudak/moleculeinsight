"""Tests for QSAR prediction pipeline advanced functionality.

Tests pipeline integration, full execution, model persistence,
new molecule prediction, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from app.qsar.qsar_prediction import QSARPipeline


class TestRunFullPipeline:
    """Test complete pipeline execution."""

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    @patch("app.qsar.qsar_prediction.get_cleaned_dataset")
    @patch("app.qsar.qsar_prediction.compute_morgan_fingerprints")
    def test_run_full_pipeline_success(self, mock_features, mock_clean, mock_load):
        """Test successful full pipeline run."""
        # Setup mocks
        raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC"] * 20,
                "value": [5.0] * 20,
            }
        )
        mock_load.return_value = {"success": True, "data": raw_data}

        cleaned_data = pd.DataFrame(
            {
                "smiles": ["CC"] * 20,
                "pIC50": np.random.randn(20),
            }
        )
        mock_clean.return_value = (cleaned_data, {})

        X = np.random.randn(20, 2048)
        mock_features.return_value = {
            "success": True,
            "X": X,
            "feature_names": [f"f_{i}" for i in range(2048)],
        }

        pipeline = QSARPipeline()

        def train_models_side_effect():
            """Simulate train_models behavior."""
            pipeline.metrics = {"rf_r2": 0.8}
            pipeline.splits = {
                "X_train": np.random.randn(15, 2048),
                "X_test": np.random.randn(5, 2048),
            }
            return {
                "success": True,
                "metrics": {"rf_r2": 0.8},
            }

        with patch.object(pipeline, "train_models", side_effect=train_models_side_effect):
            with patch.object(pipeline, "get_predictions") as mock_pred:
                with patch.object(pipeline, "get_explanations") as mock_explain:
                    mock_pred.return_value = pd.DataFrame({"pIC50": [5.0] * 20})
                    mock_explain.return_value = {"importance": []}

                    result = pipeline.run_full_pipeline()

                    assert result["success"] is True
                    assert "data" in result
                    assert "models" in result

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    def test_run_full_pipeline_load_failure(self, mock_load):
        """Test full pipeline with load failure."""
        mock_load.return_value = {"success": False, "error": "Load error"}

        pipeline = QSARPipeline()
        result = pipeline.run_full_pipeline()

        assert result["success"] is False

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    @patch("app.qsar.qsar_prediction.get_cleaned_dataset")
    def test_run_full_pipeline_preprocess_failure(self, mock_clean, mock_load):
        """Test full pipeline with preprocess failure."""
        raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC"],
                "value": [5.0],
            }
        )
        mock_load.return_value = {"success": True, "data": raw_data}
        mock_clean.return_value = (None, {})

        pipeline = QSARPipeline()
        result = pipeline.run_full_pipeline()

        assert result["success"] is False


class TestPredictNewMolecules:
    """Test new molecule prediction."""

    def test_predict_new_molecules_morgan_success(self):
        """Test successful new molecule prediction with Morgan fingerprints."""
        from sklearn.ensemble import RandomForestRegressor

        pipeline = QSARPipeline()
        pipeline.rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.random.randn(10, 2048)
        y = np.random.randn(10)
        pipeline.rf_model.fit(X, y)

        pipeline.feature_type = "morgan"
        pipeline.splits = {
            "X_test": np.random.randn(10, 2048),
            "y_test": np.random.randn(10),
        }

        smiles_list = ["CC", "CCC"]

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            X = np.random.randn(2, 2048)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": [f"f_{i}" for i in range(2048)],
            }

            result = pipeline.predict_new_molecules(smiles_list, model_type="rf")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "smiles" in result.columns
            assert "pIC50_pred" in result.columns

    def test_predict_new_molecules_rdkit_success(self):
        """Test successful new molecule prediction with RDKit descriptors."""
        pipeline = QSARPipeline()
        pipeline.rf_model = MagicMock()
        pipeline.feature_type = "rdkit"
        pipeline.splits = {
            "X_test": np.random.randn(10, 200),
            "y_test": np.random.randn(10),
        }

        smiles_list = ["CC"]

        with patch("app.qsar.qsar_prediction.compute_rdkit_descriptors") as mock_feat:
            X = np.random.randn(1, 200)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": [f"f_{i}" for i in range(200)],
            }

            with patch.object(pipeline.predictor, "predict") as mock_pred:
                mock_pred.return_value = np.array([5.0])

                result = pipeline.predict_new_molecules(
                    smiles_list, model_type="rf", feature_type="rdkit"
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1

    def test_predict_new_molecules_no_feature_type(self):
        """Test new molecule prediction with no feature type."""
        pipeline = QSARPipeline()
        result = pipeline.predict_new_molecules(["CC", "CCC"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_predict_new_molecules_no_model(self):
        """Test new molecule prediction with no trained model."""
        pipeline = QSARPipeline()
        pipeline.feature_type = "morgan"
        pipeline.splits = {"X_test": np.random.randn(10, 2048)}

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            X = np.random.randn(1, 2048)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": [f"f_{i}" for i in range(2048)],
            }

            result = pipeline.predict_new_molecules(["CC"])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_predict_new_molecules_unsupported_feature_type(self):
        """Test new molecule prediction with unsupported feature type."""
        pipeline = QSARPipeline()
        pipeline.rf_model = MagicMock()
        pipeline.feature_type = "unsupported"

        result = pipeline.predict_new_molecules(["CC"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_predict_new_molecules_featurization_failure(self):
        """Test new molecule prediction when featurization fails."""
        pipeline = QSARPipeline()
        pipeline.rf_model = MagicMock()
        pipeline.feature_type = "morgan"

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            mock_feat.return_value = {
                "success": False,
                "error": "Feature error",
            }

            result = pipeline.predict_new_molecules(["CC"])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_predict_new_molecules_no_splits_for_uncertainty(self):
        """Test new molecule prediction uses default uncertainty without splits."""
        pipeline = QSARPipeline()
        pipeline.rf_model = MagicMock()
        pipeline.feature_type = "morgan"
        pipeline.splits = None

        smiles_list = ["CC"]

        with patch("app.qsar.qsar_prediction.compute_morgan_fingerprints") as mock_feat:
            X = np.random.randn(1, 2048)
            mock_feat.return_value = {
                "success": True,
                "X": X,
                "feature_names": [f"f_{i}" for i in range(2048)],
            }

            with patch.object(pipeline.predictor, "predict") as mock_pred:
                mock_pred.return_value = np.array([5.0])

                result = pipeline.predict_new_molecules(smiles_list)

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1


class TestSaveModels:
    """Test model saving."""

    def test_save_models_success(self):
        """Test successful model saving."""
        from sklearn.ensemble import RandomForestRegressor

        pipeline = QSARPipeline()
        # Use real models instead of mocks to avoid pickling issues
        pipeline.rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        pipeline.rf_model.fit(X, y)

        pipeline.xgb_model = RandomForestRegressor(n_estimators=5, random_state=42)
        pipeline.xgb_model.fit(X, y)

        pipeline.feature_names = ["f_0", "f_1"]
        pipeline.feature_type = "morgan"
        pipeline.metrics = {"r2": 0.8}
        pipeline.features = np.random.randn(10, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.save_models(output_dir=tmpdir)

            assert result["success"] is True
            assert Path(tmpdir, "rf_model.pkl").exists()
            assert Path(tmpdir, "xgb_model.pkl").exists()
            assert Path(tmpdir, "metadata.pkl").exists()

    def test_save_models_no_trained_models(self):
        """Test saving when models not trained."""
        pipeline = QSARPipeline()
        result = pipeline.save_models()

        assert result["success"] is False
        assert "error" in result

    def test_save_models_creates_directory(self):
        """Test that save_models creates output directory."""
        from sklearn.ensemble import RandomForestRegressor

        pipeline = QSARPipeline()
        pipeline.rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        pipeline.rf_model.fit(X, y)

        pipeline.xgb_model = RandomForestRegressor(n_estimators=5, random_state=42)
        pipeline.xgb_model.fit(X, y)

        pipeline.feature_names = ["f_0"]
        pipeline.feature_type = "morgan"
        pipeline.metrics = {}
        pipeline.features = None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "models"
            result = pipeline.save_models(output_dir=str(output_dir))

            assert result["success"] is True
            assert output_dir.exists()

    def test_save_models_with_default_directory(self):
        """Test saving with default directory."""
        pipeline = QSARPipeline()
        pipeline.rf_model = MagicMock()
        pipeline.xgb_model = MagicMock()
        pipeline.feature_names = ["f_0"]
        pipeline.feature_type = "morgan"
        pipeline.metrics = {}
        pipeline.features = None

        with patch("pathlib.Path.mkdir"):
            with patch("joblib.dump"):
                result = pipeline.save_models()
                assert result["success"] is True


class TestLoadModels:
    """Test model loading."""

    def test_load_models_success(self):
        """Test successful model loading."""
        from sklearn.ensemble import RandomForestRegressor

        pipeline = QSARPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save real models
            import joblib

            rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
            X = np.random.randn(10, 5)
            y = np.random.randn(10)
            rf_model.fit(X, y)

            xgb_model = RandomForestRegressor(n_estimators=5, random_state=42)
            xgb_model.fit(X, y)

            joblib.dump(rf_model, Path(tmpdir) / "rf_model.pkl")
            joblib.dump(xgb_model, Path(tmpdir) / "xgb_model.pkl")

            metadata = {
                "feature_names": ["f_0", "f_1"],
                "feature_type": "morgan",
                "metrics": {"r2": 0.8},
            }
            joblib.dump(metadata, Path(tmpdir) / "metadata.pkl")

            result = pipeline.load_models(model_dir=tmpdir)

            assert result["success"] is True
            assert pipeline.rf_model is not None
            assert pipeline.xgb_model is not None
            assert pipeline.feature_type == "morgan"
            assert len(pipeline.feature_names) == 2

    def test_load_models_directory_not_exists(self):
        """Test loading from non-existent directory."""
        pipeline = QSARPipeline()
        result = pipeline.load_models(model_dir="/nonexistent/path")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_load_models_missing_model_files(self):
        """Test loading when model files missing."""
        pipeline = QSARPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only metadata, not models
            import joblib

            metadata = {"feature_names": ["f_0"]}
            joblib.dump(metadata, Path(tmpdir) / "metadata.pkl")

            result = pipeline.load_models(model_dir=tmpdir)

            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_load_models_no_metadata(self):
        """Test loading without metadata file."""
        from sklearn.ensemble import RandomForestRegressor

        pipeline = QSARPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib

            rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
            X = np.random.randn(10, 5)
            y = np.random.randn(10)
            rf_model.fit(X, y)

            xgb_model = RandomForestRegressor(n_estimators=5, random_state=42)
            xgb_model.fit(X, y)

            joblib.dump(rf_model, Path(tmpdir) / "rf_model.pkl")
            joblib.dump(xgb_model, Path(tmpdir) / "xgb_model.pkl")

            result = pipeline.load_models(model_dir=tmpdir)

            assert result["success"] is True
            assert result["metadata"] is None
            assert pipeline.feature_type is None  # Should be None without metadata


class TestPipelineIntegration:
    """Test complete pipeline integration."""

    def test_pipeline_state_transitions(self):
        """Test pipeline state changes through workflow."""
        pipeline = QSARPipeline()

        # Initial state
        assert pipeline.raw_data is None
        assert pipeline.cleaned_data is None
        assert pipeline.features is None

        # After loading
        raw = pd.DataFrame(
            {
                "canonical_smiles": ["CC"],
                "value": [5.0],
            }
        )
        pipeline.raw_data = raw
        assert pipeline.raw_data is not None

        # After preprocessing
        cleaned = pd.DataFrame({"smiles": ["CC"], "pIC50": [5.0]})
        pipeline.cleaned_data = cleaned
        assert pipeline.cleaned_data is not None

        # After featurization
        pipeline.features = np.random.randn(1, 10)
        pipeline.feature_type = "morgan"
        assert pipeline.features is not None
        assert pipeline.feature_type == "morgan"

    def test_pipeline_with_different_feature_types(self):
        """Test pipeline works with different feature types."""
        for ftype in ["morgan", "rdkit"]:
            pipeline = QSARPipeline()
            pipeline.cleaned_data = pd.DataFrame(
                {
                    "smiles": ["CC"],
                    "pIC50": [5.0],
                }
            )

            with patch(
                f"app.qsar.qsar_prediction.compute_{'morgan_fingerprints' if ftype == 'morgan' else 'rdkit_descriptors'}"
            ) as mock_feat:
                size = 2048 if ftype == "morgan" else 200
                X = np.random.randn(1, size)
                mock_feat.return_value = {
                    "success": True,
                    "X": X,
                    "feature_names": [f"f_{i}" for i in range(size)],
                }

                result = pipeline.featurize_data(fingerprint_type=ftype)

                assert result["success"] is True
                assert pipeline.feature_type == ftype


class TestFullPipelineFailures:
    """Test complete pipeline failure modes."""

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    @patch("app.qsar.qsar_prediction.get_cleaned_dataset")
    @patch("app.qsar.qsar_prediction.compute_morgan_fingerprints")
    def test_run_full_pipeline_featurize_failure(self, mock_features, mock_clean, mock_load):
        """Test full pipeline with featurization failure."""
        raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC"],
                "value": [5.0],
            }
        )
        mock_load.return_value = {"success": True, "data": raw_data}

        cleaned_data = pd.DataFrame({"smiles": ["CC"], "pIC50": [5.0]})
        mock_clean.return_value = (cleaned_data, {})

        mock_features.return_value = {
            "success": False,
            "error": "Feature error",
        }

        pipeline = QSARPipeline()
        result = pipeline.run_full_pipeline()

        assert result["success"] is False

    @patch("app.qsar.qsar_prediction.get_egfr_ic50_data")
    @patch("app.qsar.qsar_prediction.get_cleaned_dataset")
    @patch("app.qsar.qsar_prediction.compute_morgan_fingerprints")
    def test_run_full_pipeline_train_failure(self, mock_features, mock_clean, mock_load):
        """Test full pipeline with training failure."""
        raw_data = pd.DataFrame(
            {
                "canonical_smiles": ["CC"],
                "value": [5.0],
            }
        )
        mock_load.return_value = {"success": True, "data": raw_data}

        cleaned_data = pd.DataFrame({"smiles": ["CC"], "pIC50": [5.0]})
        mock_clean.return_value = (cleaned_data, {})

        X = np.random.randn(1, 2048)
        mock_features.return_value = {
            "success": True,
            "X": X,
            "feature_names": [f"f_{i}" for i in range(2048)],
        }

        pipeline = QSARPipeline()
        with patch.object(pipeline, "train_models") as mock_train:
            mock_train.return_value = {"success": False}
            result = pipeline.run_full_pipeline()
            assert result["success"] is False


class TestSaveModelsErrors:
    """Test model saving error handling."""

    def test_save_models_exception_handling(self):
        """Test save_models with exception."""
        from sklearn.ensemble import RandomForestRegressor

        pipeline = QSARPipeline()
        pipeline.rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        pipeline.rf_model.fit(X, y)

        pipeline.xgb_model = RandomForestRegressor(n_estimators=5, random_state=42)
        pipeline.xgb_model.fit(X, y)

        pipeline.feature_names = ["f_0"]
        pipeline.feature_type = "morgan"
        pipeline.metrics = {}
        pipeline.features = None

        # Try to save to invalid path
        with patch("joblib.dump", side_effect=Exception("Dump error")):
            result = pipeline.save_models(output_dir="/invalid/path")
            assert result["success"] is False
            assert "error" in result


class TestLoadModelsErrors:
    """Test model loading error handling."""

    def test_load_models_exception_handling(self):
        """Test load_models with exception."""
        pipeline = QSARPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib

            # Create valid model files first
            from sklearn.ensemble import RandomForestRegressor

            rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
            X = np.random.randn(10, 5)
            y = np.random.randn(10)
            rf_model.fit(X, y)

            joblib.dump(rf_model, Path(tmpdir) / "rf_model.pkl")
            joblib.dump(rf_model, Path(tmpdir) / "xgb_model.pkl")

            # Now patch joblib.load to raise an exception
            with patch("joblib.load", side_effect=Exception("Load error")):
                result = pipeline.load_models(model_dir=tmpdir)
                assert result["success"] is False
                assert "error" in result
