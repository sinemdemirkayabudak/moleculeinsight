"""Tests for QSAR model training."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from app.qsar.train import QSARModelTrainer


class TestQSARModelTrainerInit:
    """Test trainer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        trainer = QSARModelTrainer()
        assert trainer.test_size == 0.2
        assert trainer.random_state == 42
        assert isinstance(trainer.models, dict)
        assert isinstance(trainer.metrics, dict)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        trainer = QSARModelTrainer(test_size=0.3, random_state=123)
        assert trainer.test_size == 0.3
        assert trainer.random_state == 123


class TestPrepareData:
    """Test data preparation and splitting."""

    def test_prepare_data_valid_input(self):
        """Test data preparation with valid input."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        trainer = QSARModelTrainer()

        result = trainer.prepare_data(X, y)

        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert len(result["X_train"]) == 80
        assert len(result["X_test"]) == 20

    def test_prepare_data_mismatched_samples(self):
        """Test data preparation with mismatched sample counts."""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)
        trainer = QSARModelTrainer()

        with pytest.raises(ValueError, match="same n_samples"):
            trainer.prepare_data(X, y)

    def test_prepare_data_insufficient_samples(self):
        """Test data preparation with insufficient samples."""
        X = np.random.randn(5, 10)
        y = np.random.randn(5)
        trainer = QSARModelTrainer()

        with pytest.raises(ValueError, match="at least 10 samples"):
            trainer.prepare_data(X, y)

    def test_prepare_data_with_nan_values(self):
        """Test data preparation with NaN values (warning, not error)."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        y = np.random.randn(100)
        trainer = QSARModelTrainer()

        # Should not raise, but logs warning
        result = trainer.prepare_data(X, y)
        assert result is not None


class TestRandomForestTraining:
    """Test RandomForest model training."""

    def test_train_random_forest_basic(self):
        """Test basic RandomForest training."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_random_forest(X_train, y_train)

        assert model is not None
        assert hasattr(model, "predict")
        assert "rf" in trainer.models

    def test_train_random_forest_custom_estimators(self):
        """Test RandomForest with custom number of estimators."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_random_forest(X_train, y_train, n_estimators=50)

        assert model.n_estimators == 50

    def test_train_random_forest_with_cv(self):
        """Test RandomForest training with cross-validation."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_random_forest(X_train, y_train, cross_val=True, cv_folds=3)

        assert model is not None


class TestXGBoostTraining:
    """Test XGBoost model training."""

    def test_train_xgboost_basic(self):
        """Test basic XGBoost training."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_xgboost(X_train, y_train)

        assert model is not None
        assert hasattr(model, "predict")
        assert "xgb" in trainer.models

    def test_train_xgboost_custom_estimators(self):
        """Test XGBoost with custom number of estimators."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_xgboost(X_train, y_train, n_estimators=100)

        assert model is not None

    def test_train_xgboost_with_cv(self):
        """Test XGBoost training with cross-validation."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_xgboost(X_train, y_train, cross_val=True, cv_folds=3)

        assert model is not None


class TestTrainBothModels:
    """Test training both models."""

    def test_train_both_models(self):
        """Test training both RandomForest and XGBoost."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        trainer = QSARModelTrainer()

        rf_model, xgb_model, results = trainer.train_both_models(X, y)

        assert rf_model is not None
        assert xgb_model is not None
        assert "splits" in results
        assert "metrics" in results
        assert "rf" in results["metrics"]
        assert "xgb" in results["metrics"]

    def test_train_both_models_with_cv(self):
        """Test training both models with cross-validation."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        trainer = QSARModelTrainer()

        rf_model, xgb_model, results = trainer.train_both_models(X, y, cross_val=True)

        assert rf_model is not None
        assert xgb_model is not None


class TestModelPersistence:
    """Test saving and loading models."""

    def test_save_model_success(self):
        """Test successful model saving."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_random_forest(X_train, y_train)

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pkl"
            success = trainer.save_model(model, "test_model", filepath)

            assert success is True
            assert filepath.exists()

    def test_save_model_creates_directories(self):
        """Test that save_model creates parent directories."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_random_forest(X_train, y_train)

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dirs" / "test_model.pkl"
            success = trainer.save_model(model, "test_model", filepath)

            assert success is True
            assert filepath.exists()

    def test_load_model_success(self):
        """Test successful model loading."""
        X_train = np.random.randn(80, 10)
        y_train = np.random.randn(80)
        trainer = QSARModelTrainer()

        model = trainer.train_random_forest(X_train, y_train)

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pkl"
            trainer.save_model(model, "test_model", filepath)

            loaded_model = trainer.load_model(filepath, "test_model")
            assert loaded_model is not None
            assert hasattr(loaded_model, "predict")

    def test_load_model_nonexistent(self):
        """Test loading nonexistent model returns None."""
        trainer = QSARModelTrainer()
        loaded_model = trainer.load_model("/nonexistent/path/model.pkl")
        assert loaded_model is None


class TestEvaluateModel:
    """Test model evaluation."""

    def test_evaluate_model_returns_metrics(self):
        """Test that model evaluation returns valid metrics."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        trainer = QSARModelTrainer()

        splits = trainer.prepare_data(X, y)
        model = trainer.train_random_forest(splits["X_train"], splits["y_train"])

        metrics = trainer.evaluate_model(model, splits["X_test"], splits["y_test"], "RandomForest")

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert isinstance(metrics["r2"], (int, float))
        assert isinstance(metrics["rmse"], (int, float))

    def test_evaluate_model_reasonable_values(self):
        """Test that metrics have reasonable values."""
        X = np.random.randn(100, 10)
        y = X[:, 0] + 0.1 * np.random.randn(100)  # y correlated with X
        trainer = QSARModelTrainer()

        splits = trainer.prepare_data(X, y)
        model = trainer.train_random_forest(splits["X_train"], splits["y_train"])

        metrics = trainer.evaluate_model(model, splits["X_test"], splits["y_test"], "RandomForest")

        # R2 should be positive for correlated y
        assert metrics["r2"] > 0
        # RMSE should be positive
        assert metrics["rmse"] > 0


class TestTrainEdgeCases:
    """Test training error handling and edge cases."""

    def test_train_random_forest_exception_handling(self):
        """Test random forest training exception handling."""
        trainer = QSARModelTrainer()
        from unittest.mock import patch

        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)

        # Patch fit to raise an exception
        with patch(
            "sklearn.ensemble.RandomForestRegressor.fit", side_effect=Exception("Training error")
        ):
            with pytest.raises(RuntimeError):
                trainer.train_random_forest(X_train, y_train)

    def test_train_xgboost_exception_handling(self):
        """Test XGBoost training exception handling."""
        trainer = QSARModelTrainer()
        from unittest.mock import patch

        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)

        # Patch XGBRegressor to raise an exception
        with patch("xgboost.XGBRegressor.fit", side_effect=Exception("XGB error")):
            with pytest.raises(RuntimeError):
                trainer.train_xgboost(X_train, y_train)

    def test_train_random_forest_with_cross_validation(self):
        """Test random forest training with cross-validation."""
        trainer = QSARModelTrainer()

        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)

        model = trainer.train_random_forest(X_train, y_train, cross_val=True, cv_folds=3)

        assert model is not None
        assert hasattr(model, "predict")

    def test_train_both_models_exception_handling(self):
        """Test train_both_models exception handling."""
        trainer = QSARModelTrainer()
        from unittest.mock import patch

        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        splits = trainer.prepare_data(X, y)

        # Patch RandomForest training to fail
        with patch.object(trainer, "train_random_forest", side_effect=Exception("RF error")):
            with pytest.raises(RuntimeError):
                trainer.train_both_models(
                    splits["X_train"],
                    splits["y_train"],
                    splits["X_test"],
                    splits["y_test"],
                )

    def test_train_both_models_with_cv(self):
        """Test train_both_models with cross-validation."""
        trainer = QSARModelTrainer()

        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        rf_model, xgb_model, results = trainer.train_both_models(X, y, cross_val=True)

        assert rf_model is not None
        assert xgb_model is not None
        assert isinstance(results, dict)

    def test_save_model_exception_handling(self):
        """Test save_model exception handling."""
        trainer = QSARModelTrainer()
        from unittest.mock import patch

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        model.fit(X, y)

        # Patch pickle.dump to raise an exception
        with patch("pickle.dump", side_effect=Exception("Save error")):
            result = trainer.save_model(model, "test_model", "/tmp/test_model.pkl")
            assert result is False
