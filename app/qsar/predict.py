"""
Make predictions with confidence intervals on QSAR models.

Handles point predictions and uncertainty quantification.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.config import logger


class QSARPredictor:
    """Generate predictions with confidence intervals."""

    @staticmethod
    def predict(
        model: Any,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Make point predictions.

        Parameters
        ----------
        model : Any
            Trained model: RandomForest, XGBoost, etc.
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Predicted pIC50 values
        """
        return model.predict(X)

    @staticmethod
    def compute_confidence_intervals(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """
        Compute predictions with confidence intervals.

        Uses residual standard error from test set to estimate uncertainty.

        Parameters
        ----------
        model : Any
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        model_name : str
            Model identifier for logging
        ci : float
            Confidence level (default 0.95 = 95%)

        Returns
        -------
        pd.DataFrame
            Columns: y_actual, y_pred, ci_lower, ci_upper, residual
        """
        y_pred = QSARPredictor.predict(model, X_test)  # Get predictions
        residuals = y_test - y_pred
        std_error = np.std(residuals)  # Estimate standard error from residuals on test set

        # Calculate z-score for given confidence interval
        # For CI=0.95: ppf(0.975) ≈ 1.96, For CI=0.90: ppf(0.95) ≈ 1.645
        z_score = stats.norm.ppf((1 + ci) / 2)
        margin = z_score * std_error

        results = pd.DataFrame(
            {
                "y_actual": y_test,
                "y_pred": y_pred,
                "ci_lower": y_pred - margin,
                "ci_upper": y_pred + margin,
                "residual": residuals,
            }
        )

        logger.info(f"{model_name} confidence intervals computed (margin ± {margin:.3f})")

        return results

    @staticmethod
    def predict_with_uncertainty(
        model: Any,
        X_new: np.ndarray,
        uncertainty_estimate: float,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """
        Predict on new data with uncertainty band.

        Parameters
        ----------
        model : Any
            Trained model
        X_new : np.ndarray
            New feature matrix to predict on
        uncertainty_estimate : float
            Standard error from training data
        ci : float
            Confidence level (default 0.95 = 95%)

        Returns
        -------
        pd.DataFrame
            Predictions with confidence intervals
        """
        y_pred = QSARPredictor.predict(model, X_new)
        # Calculate z-score for given confidence interval
        z_score = stats.norm.ppf((1 + ci) / 2)
        margin = z_score * uncertainty_estimate

        return pd.DataFrame(
            {
                "pIC50_pred": y_pred,
                "ci_lower": y_pred - margin,
                "ci_upper": y_pred + margin,
            }
        )
