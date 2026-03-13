"""Tests for QSAR SHAP visualization."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from app.qsar.explain import QSARExplainer
from app.qsar.visualize import SHAPVisualizer


class TestValidateSavePath:
    """Test save path validation."""

    def test_validate_save_path_none(self):
        """Test that None save path is accepted."""
        # Should not raise
        SHAPVisualizer._validate_save_path(None)

    def test_validate_save_path_valid_png(self):
        """Test valid PNG path."""
        # Should not raise
        SHAPVisualizer._validate_save_path("/tmp/plot.png")

    def test_validate_save_path_valid_pdf(self):
        """Test valid PDF path."""
        # Should not raise
        SHAPVisualizer._validate_save_path("/tmp/plot.pdf")

    def test_validate_save_path_valid_svg(self):
        """Test valid SVG path."""
        # Should not raise
        SHAPVisualizer._validate_save_path("/tmp/plot.svg")

    def test_validate_save_path_valid_jpg(self):
        """Test valid JPG path."""
        # Should not raise
        SHAPVisualizer._validate_save_path("/tmp/plot.jpg")

    def test_validate_save_path_valid_jpeg(self):
        """Test valid JPEG path."""
        # Should not raise
        SHAPVisualizer._validate_save_path("/tmp/plot.jpeg")

    def test_validate_save_path_invalid_extension(self):
        """Test invalid file extension."""
        with pytest.raises(ValueError, match="Invalid save format"):
            SHAPVisualizer._validate_save_path("/tmp/plot.bmp")

    def test_validate_save_path_no_extension(self):
        """Test path with no extension."""
        with pytest.raises(ValueError, match="no file extension"):
            SHAPVisualizer._validate_save_path("/tmp/plot")

    def test_validate_save_path_case_insensitive(self):
        """Test that extension validation is case insensitive."""
        # Should not raise (uppercase extension)
        SHAPVisualizer._validate_save_path("/tmp/plot.PNG")
        SHAPVisualizer._validate_save_path("/tmp/plot.PDF")


class TestValidateBaseValue:
    """Test base value validation."""

    def test_validate_base_value_int(self):
        """Test int base value."""
        # Should not raise
        SHAPVisualizer._validate_base_value(5)

    def test_validate_base_value_float(self):
        """Test float base value."""
        # Should not raise
        SHAPVisualizer._validate_base_value(5.5)

    def test_validate_base_value_numpy_float(self):
        """Test numpy float base value."""
        # Should not raise
        SHAPVisualizer._validate_base_value(np.float64(5.5))

    def test_validate_base_value_numpy_int(self):
        """Test numpy int base value."""
        # Should not raise
        SHAPVisualizer._validate_base_value(np.int64(5))

    def test_validate_base_value_zero(self):
        """Test zero base value."""
        # Should not raise
        SHAPVisualizer._validate_base_value(0.0)

    def test_validate_base_value_negative(self):
        """Test negative base value."""
        # Should not raise
        SHAPVisualizer._validate_base_value(-5.0)

    def test_validate_base_value_string(self):
        """Test string base value raises error."""
        with pytest.raises((ValueError, TypeError)):
            SHAPVisualizer._validate_base_value("5.0")

    def test_validate_base_value_none(self):
        """Test None base value raises error."""
        with pytest.raises((ValueError, TypeError)):
            SHAPVisualizer._validate_base_value(None)

    def test_validate_base_value_bool(self):
        """Test bool base value raises error."""
        with pytest.raises((ValueError, TypeError)):
            SHAPVisualizer._validate_base_value(True)


class TestValidateAndPrepare:
    """Test data validation and preparation."""

    def test_validate_and_prepare_valid_input(self):
        """Test validation with valid input."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        feature_names = [f"feature_{i}" for i in range(10)]

        result = SHAPVisualizer._validate_and_prepare(shap_vals, X, feature_names)

        assert result is not None

    def test_validate_and_prepare_mismatched_samples(self):
        """Test validation with mismatched sample counts."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(15, 10)
        feature_names = [f"feature_{i}" for i in range(10)]

        with pytest.raises(ValueError):
            SHAPVisualizer._validate_and_prepare(shap_vals, X, feature_names)

    def test_validate_and_prepare_mismatched_features(self):
        """Test validation with mismatched feature counts."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 8)
        feature_names = [f"feature_{i}" for i in range(10)]

        with pytest.raises(ValueError):
            SHAPVisualizer._validate_and_prepare(shap_vals, X, feature_names)

    def test_validate_and_prepare_generates_feature_names(self):
        """Test that feature names are generated if not provided."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)

        result = SHAPVisualizer._validate_and_prepare(shap_vals, X, None)

        assert result is not None

    def test_validate_and_prepare_1d_array_rejection(self):
        """Test that 1D arrays are rejected."""
        shap_vals = np.random.randn(20)  # 1D
        X = np.random.randn(20, 10)
        feature_names = [f"feature_{i}" for i in range(10)]

        with pytest.raises(ValueError):
            SHAPVisualizer._validate_and_prepare(shap_vals, X, feature_names)


class TestSummaryPlot:
    """Test summary plot generation."""

    def test_summary_plot_returns_figure(self):
        """Test that summary_plot returns matplotlib figure."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        feature_names = [f"feature_{i}" for i in range(10)]
        base_value = 5.0

        fig = SHAPVisualizer.summary_plot(shap_vals, X, base_value, feature_names=feature_names)

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_with_title(self):
        """Test summary_plot with title."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        feature_names = [f"feature_{i}" for i in range(10)]
        base_value = 5.0

        fig = SHAPVisualizer.summary_plot(
            shap_vals,
            X,
            base_value,
            feature_names=feature_names,
            title="Test Summary Plot",
        )

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_with_save_path(self):
        """Test summary_plot with file save."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        feature_names = [f"feature_{i}" for i in range(10)]
        base_value = 5.0

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "summary.png"

            fig = SHAPVisualizer.summary_plot(
                shap_vals,
                X,
                base_value,
                feature_names=feature_names,
                save_path=str(save_path),
            )

            assert fig is not None
            assert save_path.exists()
            plt.close(fig)

    def test_summary_plot_with_max_display(self):
        """Test summary_plot with max_display parameter."""
        shap_vals = np.random.randn(20, 50)
        X = np.random.randn(20, 50)
        base_value = 5.0

        fig = SHAPVisualizer.summary_plot(shap_vals, X, base_value, max_display=10)

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_invalid_save_path(self):
        """Test summary_plot with invalid save path."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        with pytest.raises(ValueError):
            SHAPVisualizer.summary_plot(shap_vals, X, base_value, save_path="/tmp/plot.bmp")


class TestDependencePlot:
    """Test dependence plot generation."""

    def test_dependence_plot_returns_figure(self):
        """Test that dependence_plot returns matplotlib figure."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        feature_names = [f"feature_{i}" for i in range(10)]

        fig = SHAPVisualizer.dependence_plot(
            shap_vals, X, feature_idx=0, feature_names=feature_names
        )

        assert fig is not None
        plt.close(fig)

    def test_dependence_plot_with_title(self):
        """Test dependence_plot with title."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)

        fig = SHAPVisualizer.dependence_plot(
            shap_vals, X, feature_idx=0, title="Test Dependence Plot"
        )

        assert fig is not None
        plt.close(fig)

    def test_dependence_plot_with_save_path(self):
        """Test dependence_plot with file save."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "dependence.png"

            fig = SHAPVisualizer.dependence_plot(
                shap_vals, X, feature_idx=0, save_path=str(save_path)
            )

            assert fig is not None
            assert save_path.exists()
            plt.close(fig)

    def test_dependence_plot_different_features(self):
        """Test dependence_plot for different feature indices."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)

        for idx in [0, 1, 5, 9]:
            fig = SHAPVisualizer.dependence_plot(shap_vals, X, feature_idx=idx)
            assert fig is not None
            plt.close(fig)

    def test_dependence_plot_invalid_index(self):
        """Test dependence_plot with invalid feature index."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)

        with pytest.raises(ValueError):
            SHAPVisualizer.dependence_plot(shap_vals, X, feature_idx=20)


class TestForcePlot:
    """Test force plot generation."""

    def test_force_plot_returns_figure(self):
        """Test that force_plot returns matplotlib figure."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        fig = SHAPVisualizer.force_plot(shap_vals, X, base_value, sample_idx=0)

        assert fig is not None
        plt.close(fig)

    def test_force_plot_with_title(self):
        """Test force_plot with title."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        fig = SHAPVisualizer.force_plot(
            shap_vals, X, base_value, sample_idx=0, title="Test Force Plot"
        )

        assert fig is not None
        plt.close(fig)

    def test_force_plot_with_save_path(self):
        """Test force_plot with file save."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "force.png"

            fig = SHAPVisualizer.force_plot(
                shap_vals,
                X,
                base_value,
                sample_idx=0,
                save_path=str(save_path),
            )

            assert fig is not None
            assert save_path.exists()
            plt.close(fig)

    def test_force_plot_different_samples(self):
        """Test force_plot for different samples."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        for idx in [0, 5, 19]:
            fig = SHAPVisualizer.force_plot(shap_vals, X, base_value, sample_idx=idx)
            assert fig is not None
            plt.close(fig)

    def test_force_plot_with_figsize(self):
        """Test force_plot with custom figsize."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        fig = SHAPVisualizer.force_plot(shap_vals, X, base_value, figsize=(16, 4))

        assert fig is not None
        plt.close(fig)


class TestHeatmapPlot:
    """Test heatmap plot generation."""

    def test_heatmap_returns_figure(self):
        """Test that heatmap returns matplotlib figure."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        fig = SHAPVisualizer.heatmap(shap_vals, X, base_value)

        assert fig is not None
        plt.close(fig)

    def test_heatmap_with_title(self):
        """Test heatmap with title."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        fig = SHAPVisualizer.heatmap(shap_vals, X, base_value, title="Test Heatmap")

        assert fig is not None
        plt.close(fig)

    def test_heatmap_with_save_path(self):
        """Test heatmap with file save."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "heatmap.png"

            fig = SHAPVisualizer.heatmap(shap_vals, X, base_value, save_path=str(save_path))

            assert fig is not None
            assert save_path.exists()
            plt.close(fig)

    def test_heatmap_with_max_display(self):
        """Test heatmap with max_display parameter."""
        shap_vals = np.random.randn(100, 50)
        X = np.random.randn(100, 50)
        base_value = 5.0

        fig = SHAPVisualizer.heatmap(shap_vals, X, base_value, max_display=20)

        assert fig is not None
        plt.close(fig)

    def test_heatmap_with_feature_names(self):
        """Test heatmap with feature names."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0
        feature_names = [f"feature_{i}" for i in range(10)]

        fig = SHAPVisualizer.heatmap(shap_vals, X, base_value, feature_names=feature_names)

        assert fig is not None
        plt.close(fig)


class TestValidateAndPrepareEdgeCases:
    """Test validation edge cases for complete coverage."""

    def test_validate_shap_vals_scalar(self):
        """Test that scalar SHAP values are rejected."""
        shap_scalar = np.array(5.0)  # 0D array
        X = np.random.randn(1, 10)

        with pytest.raises(ValueError, match="scalar"):
            SHAPVisualizer._validate_and_prepare(shap_scalar, X, None)

    def test_validate_shap_vals_3d(self):
        """Test that 3D SHAP values are rejected."""
        shap_3d = np.random.randn(5, 10, 2)  # 3D array
        X = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            SHAPVisualizer._validate_and_prepare(shap_3d, X, None)

    def test_validate_x_scalar(self):
        """Test that scalar X is rejected."""
        shap_vals = np.random.randn(1, 10)
        X_scalar = np.array(5.0)  # 0D array

        with pytest.raises(ValueError, match="scalar"):
            SHAPVisualizer._validate_and_prepare(shap_vals, X_scalar, None)

    def test_validate_x_3d(self):
        """Test that 3D X is rejected."""
        shap_vals = np.random.randn(5, 10)
        X_3d = np.random.randn(5, 10, 2)  # 3D array

        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            SHAPVisualizer._validate_and_prepare(shap_vals, X_3d, None)

    def test_validate_x_1d(self):
        """Test that 1D X is normalized to 2D."""
        shap_vals = np.random.randn(10)
        X_1d = np.random.randn(10)  # 1D array

        shap_vals_out, X_out, _ = SHAPVisualizer._validate_and_prepare(shap_vals, X_1d, None)

        assert shap_vals_out.ndim == 2
        assert X_out.ndim == 2

    def test_validate_feature_name_mismatch(self):
        """Test validation with mismatched feature name count."""
        shap_vals = np.random.randn(5, 10)
        X = np.random.randn(5, 10)
        feature_names = [f"f_{i}" for i in range(8)]  # Wrong number

        with pytest.raises(ValueError, match="Feature names"):
            SHAPVisualizer._validate_and_prepare(shap_vals, X, feature_names)

    def test_validate_force_plot_sample_out_of_range(self):
        """Test force_plot with invalid sample index."""
        shap_vals = np.random.randn(5, 10)
        X = np.random.randn(5, 10)
        base_value = 5.0

        with pytest.raises(ValueError, match="Sample index"):
            SHAPVisualizer.force_plot(shap_vals, X, base_value, sample_idx=10)

    def test_validate_dependence_plot_invalid_feature_idx(self):
        """Test dependence_plot with feature index out of bounds."""
        shap_vals = np.random.randn(5, 10)
        X = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Feature index"):
            SHAPVisualizer.dependence_plot(shap_vals, X, feature_idx=15)


class TestVisualizationIntegration:
    """Test end-to-end visualization workflow."""

    def test_full_visualization_pipeline(self):
        """Test complete visualization from model to plots."""
        # Train model
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Generate SHAP values
        explainer_obj = QSARExplainer()
        explainer = explainer_obj.create_explainer(model, X_train)
        X_test = np.random.randn(20, 10)
        shap_vals = explainer_obj.compute_shap_values(explainer, X_test)

        # Generate visualizations
        base_value = 5.0

        fig1 = SHAPVisualizer.summary_plot(shap_vals, X_test, base_value)
        fig2 = SHAPVisualizer.dependence_plot(shap_vals, X_test, feature_idx=0)
        fig3 = SHAPVisualizer.force_plot(shap_vals, X_test, base_value)
        fig4 = SHAPVisualizer.heatmap(shap_vals, X_test, base_value)

        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
        assert fig4 is not None

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

    def test_all_formats_save(self):
        """Test saving to all supported formats."""
        shap_vals = np.random.randn(20, 10)
        X = np.random.randn(20, 10)
        base_value = 5.0
        formats = ["png", "pdf", "svg", "jpg", "jpeg"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in formats:
                save_path = Path(tmpdir) / f"plot.{fmt}"
                fig = SHAPVisualizer.summary_plot(
                    shap_vals, X, base_value, save_path=str(save_path)
                )
                assert save_path.exists()
                plt.close(fig)
