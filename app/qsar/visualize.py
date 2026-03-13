"""
SHAP visualization for QSAR model explanations.

Separate functions for summary, force, dependence, and heatmap plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

from app.config import logger

# Valid image formats for saving plots
VALID_FORMATS = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}


class SHAPVisualizer:
    """Static methods for SHAP-based model visualization."""

    @staticmethod
    def _validate_save_path(save_path: str | None) -> None:
        """Validate save_path has valid image format extension.

        Args:
            save_path: Path to save figure to

        Raises:
            ValueError: If file extension not in VALID_FORMATS
        """
        if save_path is None:
            return

        ext = Path(save_path).suffix.lower()
        if not ext:
            raise ValueError(
                f"Save path '{save_path}' has no file extension. "
                f"Choose from {', '.join(sorted(VALID_FORMATS))}"
            )

        if ext not in VALID_FORMATS:
            raise ValueError(
                f"Invalid save format '{ext}' in '{save_path}'. "
                f"Choose from {', '.join(sorted(VALID_FORMATS))}"
            )

    @staticmethod
    def _validate_base_value(base_value: float) -> None:
        """Validate base_value is a numeric type.

        Args:
            base_value: Model baseline value

        Raises:
            TypeError: If base_value is not int, float, or numpy numeric type
        """
        if not isinstance(base_value, (int, float, np.floating, np.integer)) or isinstance(
            base_value, bool
        ):
            raise TypeError(f"base_value must be a number, got {type(base_value).__name__}")

    @staticmethod
    def _validate_and_prepare(
        shap_vals: np.ndarray,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Validate SHAP values, feature matrix, and feature names."""
        # Validate shap_vals
        if shap_vals.ndim == 0:
            error_msg = "SHAP values cannot be a scalar (0D array)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if shap_vals.ndim > 2:
            error_msg = f"Expected 1D or 2D SHAP values, got {shap_vals.ndim}D"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize shap_vals to 2D
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)

        # Validate X
        if X.ndim == 0:
            error_msg = "Feature matrix X cannot be a scalar (0D array)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if X.ndim > 2:
            error_msg = f"Expected 1D or 2D feature matrix, got {X.ndim}D"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize X to 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Check shapes match
        if shap_vals.shape[0] != X.shape[0]:
            error_msg = (
                f"SHAP values ({shap_vals.shape[0]} samples) != features ({X.shape[0]} samples)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if shap_vals.shape[1] != X.shape[1]:
            error_msg = f"SHAP features ({shap_vals.shape[1]}) != X features ({X.shape[1]})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_features = shap_vals.shape[1]

        # Validate feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            error_msg = f"Feature names ({len(feature_names)}) != features ({n_features})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return shap_vals, X, feature_names

    @staticmethod
    def summary_plot(
        shap_vals: np.ndarray,
        X: np.ndarray,
        base_value: float,
        feature_names: list[str] | None = None,
        max_display: int = 20,
        figsize: tuple[int, int] = (12, 8),
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create SHAP summary plot showing feature importance and direction.

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values (n_samples, n_features)
        X : np.ndarray
            Feature matrix for coloring
        base_value : float
            Base value / expected value (required - usually mean prediction). Essential for correct SHAP interpretation.
        feature_names : List[str], optional
            Feature names (default: feature_0, feature_1, ...)
        max_display : int
            Maximum features to display (default 20)
        figsize : tuple
            Figure size (default (12, 8))
        title : str, optional
            Plot title for identifying model/data
        save_path : str, optional
            Path to save PNG

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        SHAPVisualizer._validate_save_path(save_path)
        SHAPVisualizer._validate_base_value(base_value)

        shap_vals, X, feature_names = SHAPVisualizer._validate_and_prepare(
            shap_vals, X, feature_names
        )

        plt.figure(figsize=figsize)
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=np.full(len(shap_vals), base_value),
            data=X,
            feature_names=feature_names,
        )
        shap.summary_plot(explanation, plot_type="dot", max_display=max_display, show=False)
        fig = plt.gcf()  # Capture active figure AFTER SHAP plotting

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        logger.info("Summary plot created")
        return fig

    @staticmethod
    def dependence_plot(
        shap_vals: np.ndarray,
        X: np.ndarray,
        feature_idx: int = 0,
        feature_names: list[str] | None = None,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create SHAP dependence plot for individual feature.

        Shows how a single feature's SHAP values change as the feature value
        changes. Color of points indicates potentially interacting features.

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values (n_samples, n_features)
        X : np.ndarray
            Feature matrix
        feature_idx : int
            Feature index to plot (default 0)
        feature_names : List[str], optional
            Feature names
        figsize : tuple
            Figure size (default (10, 6))
        title : str, optional
            Plot title for identifying model/data
        save_path : str, optional
            Path to save PNG

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        SHAPVisualizer._validate_save_path(save_path)

        shap_vals, X, feature_names = SHAPVisualizer._validate_and_prepare(
            shap_vals, X, feature_names
        )

        if not 0 <= feature_idx < len(feature_names):
            raise ValueError(
                f"Feature index {feature_idx} out of valid range [0, {len(feature_names) - 1}]"
            )

        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_idx,
            shap_vals,
            X,
            feature_names=feature_names,
            show=False,
        )
        fig = plt.gcf()  # Capture active figure AFTER SHAP plotting

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Dependence plot saved to {save_path}")

        logger.info(f"Dependence plot created for {feature_names[feature_idx]}")
        return fig

    @staticmethod
    def force_plot(
        shap_vals: np.ndarray,
        X: np.ndarray,
        base_value: float,
        sample_idx: int = 0,
        feature_names: list[str] | None = None,
        figsize: tuple[int, int] = (14, 3),
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create SHAP force plot for individual prediction.

        Shows how each feature pushes prediction away from base_value.

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values (n_samples, n_features)
        X : np.ndarray
            Feature matrix
        base_value : float
            Base value / expected value (required, usually mean prediction)
        sample_idx : int
            Sample index (default 0)
        feature_names : List[str], optional
            Feature names
        figsize : tuple
            Figure size (default (14, 3))
        title : str, optional
            Plot title for identifying model/data
        save_path : str, optional
            Path to save PNG

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        SHAPVisualizer._validate_save_path(save_path)
        SHAPVisualizer._validate_base_value(base_value)

        shap_vals, X, feature_names = SHAPVisualizer._validate_and_prepare(
            shap_vals, X, feature_names
        )

        if not 0 <= sample_idx < len(shap_vals):
            raise ValueError(
                f"Sample index {sample_idx} out of valid range [0, {len(shap_vals) - 1}]"
            )

        plt.figure(figsize=figsize)
        shap.force_plot(
            base_value,
            shap_vals[sample_idx],
            X[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        fig = plt.gcf()  # Capture active figure AFTER SHAP plotting

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Force plot saved to {save_path}")

        logger.info(f"Force plot created for sample {sample_idx} (base_value={base_value})")
        return fig

    @staticmethod
    def heatmap(
        shap_vals: np.ndarray,
        X: np.ndarray,
        base_value: float,
        feature_names: list[str] | None = None,
        max_display: int = 20,
        figsize: tuple[int, int] = (14, 8),
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create SHAP heatmap showing values across samples and features.

        Uses SHAP's built-in heatmap visualization showing sample-level
        SHAP values with color-coded magnitude.

        Parameters
        ----------
        shap_vals : np.ndarray
            SHAP values (n_samples, n_features)
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        base_value : float
            Base value / expected value (required - usually mean prediction). Essential for correct SHAP interpretation.
        feature_names : List[str], optional
            Feature names (default: feature_0, feature_1, ...)
        max_display : int
            Maximum features to display (default 20)
        figsize : tuple
            Figure size (default (14, 8))
        title : str, optional
            Plot title for identifying model/data
        save_path : str, optional
            Path to save PNG

        Returns
        -------
        plt.Figure
            Matplotlib figure with heatmap
        """
        SHAPVisualizer._validate_save_path(save_path)
        SHAPVisualizer._validate_base_value(base_value)

        shap_vals, X, feature_names = SHAPVisualizer._validate_and_prepare(
            shap_vals, X, feature_names
        )

        plt.figure(figsize=figsize)
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=np.full(len(shap_vals), base_value),
            data=X,
            feature_names=feature_names,
        )
        shap.plots.heatmap(explanation, max_display=max_display, show=False)
        fig = plt.gcf()  # Capture active figure AFTER SHAP creates it

        if title:
            # Add title closer to the heatmap
            fig.text(
                0.5,
                0.75,
                title,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="top",
                transform=fig.transFigure,
            )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Heatmap saved to {save_path}")

        logger.info(f"Heatmap created with {min(max_display, len(feature_names))} features")
        return fig
