"""Tests for QSAR model explanation with SHAP."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from app.qsar.explain import QSARExplainer


class TestQSARExplainerInit:
    """Test explainer initialization."""

    def test_init(self):
        """Test explainer initialization."""
        explainer = QSARExplainer()
        assert isinstance(explainer.explainers, dict)
        assert isinstance(explainer.shap_values, dict)
        assert len(explainer.explainers) == 0
        assert len(explainer.shap_values) == 0


class TestCreateExplainer:
    """Test SHAP explainer creation."""

    def test_create_explainer_success(self):
        """Test successful explainer creation."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train, model_name="test_model")

        assert explainer is not None
        assert "test_model" in explainer_obj.explainers

    def test_create_explainer_empty_background(self):
        """Test explainer creation fails with empty background."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        X_empty = np.array([])

        with pytest.raises(ValueError, match="cannot be None or empty"):
            explainer_obj.create_explainer(model, X_empty)

    def test_create_explainer_none_background(self):
        """Test explainer creation fails with None background."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()

        with pytest.raises(ValueError, match="cannot be None or empty"):
            explainer_obj.create_explainer(model, None)

    def test_create_explainer_with_different_model_names(self):
        """Test creating multiple explainers with different names."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)

        model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        model1.fit(X_train, y_train)

        model2 = RandomForestRegressor(n_estimators=20, random_state=42)
        model2.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer_obj.create_explainer(model1, X_train, model_name="model1")
        explainer_obj.create_explainer(model2, X_train, model_name="model2")

        assert "model1" in explainer_obj.explainers
        assert "model2" in explainer_obj.explainers
        assert len(explainer_obj.explainers) == 2


class TestComputeSHAPValues:
    """Test SHAP value computation."""

    def test_compute_shap_values_returns_array(self):
        """Test that SHAP values computation returns array."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        assert isinstance(shap_vals, np.ndarray)

    def test_compute_shap_values_shape(self):
        """Test SHAP values have correct shape."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        # SHAP values should have same shape as input features
        assert shap_vals.shape[0] <= 20  # May be subsampled
        assert shap_vals.shape[1] == 10  # Same number of features

    def test_compute_shap_values_with_max_samples(self):
        """Test SHAP computation with max_samples limit."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(1000, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test, max_samples=100)

        # Should be subsampled to max_samples
        assert shap_vals.shape[0] <= 100

    def test_compute_shap_values_small_dataset(self):
        """Test SHAP computation on small dataset (no subsampling)."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test, max_samples=1000)

        # Should not be subsampled
        assert shap_vals.shape[0] == 20


class TestGetFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance_returns_dataframe(self):
        """Test feature importance returns DataFrame."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        importance = explainer_obj.get_feature_importance(shap_vals)

        assert importance is not None

    def test_get_feature_importance_with_feature_names(self):
        """Test feature importance with custom feature names."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        feature_names = [f"feature_{i}" for i in range(10)]
        importance = explainer_obj.get_feature_importance(shap_vals, feature_names=feature_names)

        assert importance is not None

    def test_get_feature_importance_top_n(self):
        """Test feature importance with top_n limit."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        importance = explainer_obj.get_feature_importance(shap_vals, top_n=5)

        assert importance is not None


class TestSummarySummaryStats:
    """Test summary statistics computation."""

    def test_summary_stats_returns_dict(self):
        """Test summary stats returns dictionary."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        stats = explainer_obj.summary_stats(shap_vals)

        assert isinstance(stats, pd.DataFrame)

    def test_summary_stats_has_required_keys(self):
        """Test summary stats has expected keys."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        stats = explainer_obj.summary_stats(shap_vals)

        # Should have at least mean values per feature
        assert len(stats) > 0

    def test_summary_stats_with_feature_names(self):
        """Test summary stats with custom feature names."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        feature_names = [f"feature_{i}" for i in range(10)]
        stats = explainer_obj.summary_stats(shap_vals, feature_names=feature_names)

        assert isinstance(stats, pd.DataFrame)


class TestExplainerEdgeCases:
    """Test edge cases in explanation."""

    def test_explain_single_sample(self):
        """Test explanation for single sample."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(1, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        assert shap_vals.shape[0] >= 1

    def test_explain_many_features(self):
        """Test explanation with many features."""
        X_train = np.random.randn(50, 100)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(5, 100)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        assert shap_vals.shape[1] == 100


class TestExplainerErrorHandling:
    """Test explainer error handling and edge cases."""

    def test_compute_shap_invalid_max_samples(self):
        """Test compute_shap_values with invalid max_samples."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="max_samples must be > 0"):
            explainer_obj.compute_shap_values(explainer, X_test, max_samples=0, model_name="test")

    def test_compute_shap_negative_max_samples(self):
        """Test compute_shap_values with negative max_samples."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)

        X_test = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="max_samples must be > 0"):
            explainer_obj.compute_shap_values(explainer, X_test, max_samples=-5, model_name="test")

    def test_get_feature_importance_1d_shap_values(self):
        """Test get_feature_importance with 1D SHAP values."""
        explainer_obj = QSARExplainer()
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer_obj.create_explainer(model, X_train)

        # Manually create 1D SHAP values
        shap_vals_1d = np.random.randn(50)

        with pytest.raises(ValueError, match="Expected 2D SHAP values"):
            explainer_obj.get_feature_importance(
                shap_vals_1d, feature_names=[f"f_{i}" for i in range(10)]
            )

    def test_get_feature_importance_feature_name_mismatch(self):
        """Test get_feature_importance with mismatched feature names."""
        explainer_obj = QSARExplainer()
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(10, 10)
        explainer = explainer_obj.create_explainer(model, X_train)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        # Wrong number of feature names
        with pytest.raises(ValueError, match="Feature names"):
            explainer_obj.get_feature_importance(
                shap_vals, feature_names=[f"f_{i}" for i in range(5)]
            )

    def test_summary_stats_1d_shap_values(self):
        """Test summary_stats with 1D SHAP values."""
        explainer_obj = QSARExplainer()
        shap_vals_1d = np.random.randn(50)

        with pytest.raises(ValueError, match="Expected 2D SHAP values"):
            explainer_obj.summary_stats(shap_vals_1d)

    def test_summary_stats_feature_name_mismatch(self):
        """Test summary_stats with mismatched feature names."""
        explainer_obj = QSARExplainer()
        shap_vals = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="Feature names"):
            explainer_obj.summary_stats(shap_vals, feature_names=[f"f_{i}" for i in range(5)])
