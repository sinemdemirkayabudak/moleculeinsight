"""
SHAP explanations for QSAR model predictions.

Generates feature importance charts and force plots.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap

from app.config import logger


class QSARExplainer:
    """Generate SHAP-based explanations for QSAR predictions."""

    def __init__(self):
        """Initialize explainer storage."""
        self.explainers = {}
        self.shap_values = {}

    def create_explainer(
        self,
        model: Any,
        X_background: np.ndarray,
        model_name: str = "model",
    ) -> shap.TreeExplainer:
        """
        Create SHAP TreeExplainer with background data.

        For tree models (RandomForest, XGBoost), background data enables
        proper baseline calculations and consistency in explanations.
        Should be a representative sample of training data.

        Parameters
        ----------
        model : Any
            Trained tree-based model (RF or XGBoost)
        X_background : np.ndarray
            Background data for SHAP baseline. Should be 100-1000 representative
            samples from training data. Critical for consistent explanations.
        model_name : str
            Model identifier

        Returns
        -------
        shap.TreeExplainer
            SHAP explainer instance

        Raises
        ------
        ValueError
            If X_background is empty or invalid
        """
        if X_background is None or len(X_background) == 0:
            error_msg = "X_background cannot be None or empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Creating SHAP explainer for {model_name} with {len(X_background)} background samples"
        )

        # TreeExplainer for tree models (RF, XGBoost) with background data
        # Background data provides proper baseline for SHAP explanations
        explainer = shap.TreeExplainer(model, data=X_background)

        self.explainers[model_name] = explainer

        logger.info(f"TreeExplainer ready for {model_name}")

        return explainer

    def compute_shap_values(
        self,
        explainer: shap.TreeExplainer,
        X: np.ndarray,
        model_name: str = "model",
        max_samples: int = 1000,
    ) -> np.ndarray:
        """
        Compute SHAP values for predictions with safe sampling.

        For large datasets, automatically subsamples to 1000 samples to prevent
        memory overflow and excessive computation. SHAP value rankings remain
        reliable even with sampling.

        Parameters
        ----------
        explainer : shap.TreeExplainer
            SHAP explainer instance
        X : np.ndarray
            Feature matrix
        model_name : str
            Model identifier
        max_samples : int
            Maximum samples to compute SHAP for (default 1000).
            Prevents memory issues on large datasets.
            Set to None for unlimited (not recommended for >10k samples).

        Returns
        -------
        np.ndarray
            SHAP values (n_samples, n_features)

        Raises
        ------
        ValueError
            If max_samples is invalid
        """
        if max_samples is not None and max_samples <= 0:
            error_msg = f"max_samples must be > 0, got {max_samples}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Use sampling for efficiency and safety on large datasets
        if max_samples is not None and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            logger.warning(
                f"Computing SHAP values for {model_name}: "
                f"subsampling from {len(X)} → {max_samples} samples "
                f"(safety limit to prevent memory overflow)"
            )
        else:
            X_sample = X
            logger.info(f"Computing SHAP values for {model_name} ({len(X)} samples)")

        shap_vals = explainer.shap_values(X_sample)
        self.shap_values[model_name] = shap_vals

        logger.info(f"SHAP computation complete for {model_name}: shape={shap_vals.shape}")

        return shap_vals

    def _validate_and_prepare_features(
        self,
        shap_vals: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Validate SHAP values and prepare feature names (internal helper).

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values to validate
        feature_names : list[str], optional
            Feature names to validate

        Returns
        -------
        tuple
            (validated_shap_vals, validated_feature_names)

        Raises
        ------
        ValueError
            If validation fails
        """
        if shap_vals.ndim != 2:
            error_msg = f"Expected 2D SHAP values, got shape {shap_vals.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_features = shap_vals.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            error_msg = f"Feature names ({len(feature_names)}) != features ({n_features})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return shap_vals, feature_names

    def get_feature_importance(
        self,
        shap_vals: np.ndarray,
        feature_names: list[str] | None = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Extract top N most important features from SHAP values.

        Ranks features by mean absolute SHAP value across all predictions.
        Measures how much each feature impacts model predictions on average.

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values (n_samples, n_features)
        feature_names : list[str], optional
            Feature names (e.g., bit_0, bit_1, ...)
        top_n : int
            Number of top features to return (default 20)

        Returns
        -------
        pd.DataFrame
            Feature importance ranked by mean absolute SHAP value
        """
        shap_vals, feature_names = self._validate_and_prepare_features(shap_vals, feature_names)

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": mean_abs_shap,
            }
        ).sort_values("importance", ascending=False)

        logger.info(f"Top {min(top_n, len(importance_df))} features extracted")

        return importance_df.head(top_n)

    def summary_stats(
        self,
        shap_vals: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate summary statistics from SHAP values.

        Computes mean, std, and max absolute SHAP values per feature.
        Shows both typical impact (mean) and variability (std) across predictions.

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values (n_samples, n_features)
        feature_names : list[str], optional
            Feature names

        Returns
        -------
        pd.DataFrame
            SHAP statistics ranked by mean importance
        """
        shap_vals, feature_names = self._validate_and_prepare_features(shap_vals, feature_names)

        mean_shap = np.abs(shap_vals).mean(axis=0)
        std_shap = np.abs(shap_vals).std(axis=0)
        max_shap = np.abs(shap_vals).max(axis=0)

        stats_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_shap": mean_shap,
                "std_shap": std_shap,
                "max_shap": max_shap,
            }
        ).sort_values("mean_shap", ascending=False)

        logger.info(f"SHAP summary statistics computed: {len(stats_df)} features")

        return stats_df
