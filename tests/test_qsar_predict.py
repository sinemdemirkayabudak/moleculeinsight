"""Tests for QSAR model prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.qsar.predict import QSARPredictor


class TestPredict:
    """Test basic prediction."""

    def test_predict_returns_array(self):
        """Test that predict returns numpy array."""
        # Create simple linear model
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 3, 5])
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.array([[2, 3], [4, 5]])
        predictions = QSARPredictor.predict(model, X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2

    def test_predict_shape_matches_input(self):
        """Test that predictions have same length as input."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        predictions = QSARPredictor.predict(model, X_test)

        assert len(predictions) == len(X_test)

    def test_predict_consistency(self):
        """Test that predictions are consistent."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(5, 10)

        pred1 = QSARPredictor.predict(model, X_test)
        pred2 = QSARPredictor.predict(model, X_test)

        np.testing.assert_array_equal(pred1, pred2)


class TestComputeConfidenceIntervals:
    """Test confidence interval computation."""

    def test_ci_returns_dataframe(self):
        """Test that CI computation returns DataFrame."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)

        result = QSARPredictor.compute_confidence_intervals(
            model, X_test, y_test, model_name="test_model"
        )

        assert isinstance(result, pd.DataFrame)

    def test_ci_dataframe_columns(self):
        """Test that CI DataFrame has required columns."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)

        result = QSARPredictor.compute_confidence_intervals(model, X_test, y_test)

        expected_cols = ["y_actual", "y_pred", "ci_lower", "ci_upper", "residual"]
        for col in expected_cols:
            assert col in result.columns

    def test_ci_bounds_valid(self):
        """Test that confidence interval bounds are valid."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)

        result = QSARPredictor.compute_confidence_intervals(model, X_test, y_test, ci=0.95)

        # Predictions should be within bounds
        assert np.all(result["y_pred"] >= result["ci_lower"])
        assert np.all(result["y_pred"] <= result["ci_upper"])

    def test_ci_lower_ci(self):
        """Test CI with lower confidence level."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)

        result_90 = QSARPredictor.compute_confidence_intervals(model, X_test, y_test, ci=0.90)
        result_95 = QSARPredictor.compute_confidence_intervals(model, X_test, y_test, ci=0.95)

        # 95% CI should be wider than 90% CI
        margin_90 = (result_90["ci_upper"] - result_90["ci_lower"]).mean()
        margin_95 = (result_95["ci_upper"] - result_95["ci_lower"]).mean()
        assert margin_95 > margin_90

    def test_ci_length_matches_data(self):
        """Test that CI results have correct length."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)

        result = QSARPredictor.compute_confidence_intervals(model, X_test, y_test)

        assert len(result) == len(X_test)
        assert len(result) == len(y_test)

    def test_ci_residuals_correct(self):
        """Test that residuals are correctly computed."""
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([1.0, 3.0, 5.0, 7.0])
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([1.0, 3.0])

        result = QSARPredictor.compute_confidence_intervals(model, X_test, y_test)

        # residual = y_actual - y_pred
        expected_residuals = y_test - result["y_pred"].values
        np.testing.assert_array_almost_equal(result["residual"].values, expected_residuals)


class TestPredictWithUncertainty:
    """Test prediction with uncertainty quantification."""

    def test_predict_with_uncertainty_returns_dict(self):
        """Test that predict_with_uncertainty returns dict."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_new = np.random.randn(5, 10)
        uncertainty_estimate = 0.5

        result = QSARPredictor.predict_with_uncertainty(model, X_new, uncertainty_estimate)

        assert isinstance(result, pd.DataFrame)

    def test_predict_with_uncertainty_keys(self):
        """Test that result has required keys."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_new = np.random.randn(5, 10)

        result = QSARPredictor.predict_with_uncertainty(model, X_new, uncertainty_estimate=0.5)

        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns

    def test_predict_with_uncertainty_bounds(self):
        """Test that uncertainty bounds are valid."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_new = np.random.randn(5, 10)

        result = QSARPredictor.predict_with_uncertainty(model, X_new, uncertainty_estimate=0.5)

        assert len(result) == len(X_new)
        assert result["ci_upper"].ge(result["ci_lower"]).all()


class TestPredictionEdgeCases:
    """Test edge cases in prediction."""

    def test_predict_single_sample(self):
        """Test prediction on single sample."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(1, 10)
        predictions = QSARPredictor.predict(model, X_test)

        assert len(predictions) == 1

    def test_predict_large_batch(self):
        """Test prediction on large batch."""
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(1000, 10)
        predictions = QSARPredictor.predict(model, X_test)

        assert len(predictions) == 1000

    def test_ci_with_perfect_predictions(self):
        """Test CI when model makes perfect predictions."""
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y_train = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        X_test = X_train
        y_test = y_train

        result = QSARPredictor.compute_confidence_intervals(model, X_test, y_test)

        # When residuals are small, bounds should be close together
        margin = (result["ci_upper"] - result["ci_lower"]).mean()
        # Margin should be non-negative
        assert margin >= 0
