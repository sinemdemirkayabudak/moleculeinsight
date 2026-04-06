"""Focused tests for similarity_search visualization - structure image generation.

Tests cover:
- create_structure_image: wrapper function
- _generate_cached_structure_image: image generation, caching, and error handling
- visualize_distribution: function logic coverage
- prepare_csv_export: CSV formatting
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

from app.similarity_search.visualization import (
    _generate_cached_structure_image,
    create_structure_image,
    prepare_csv_export,
    visualize_distribution,
)

# Mock streamlit before importing visualization
sys.modules["streamlit"] = Mock()


class TestCreateStructureImage:
    """Test structure image creation wrapper."""

    @patch("app.similarity_search.visualization._generate_cached_structure_image")
    def test_wrapper_delegates_to_cached_function(self, mock_cached):
        """Test that wrapper delegates to cached function."""
        mock_cached.return_value = "base64encodedstring"

        result = create_structure_image("q1", "CCO", "r1", "CC")

        assert result == "base64encodedstring"
        mock_cached.assert_called_once_with("q1", "CCO", "r1", "CC")

    @patch("app.similarity_search.visualization._generate_cached_structure_image")
    def test_wrapper_returns_none_on_error(self, mock_cached):
        """Test wrapper returns None when cached function returns None."""
        mock_cached.return_value = None

        result = create_structure_image("q1", "INVALID", "r1", "CC")

        assert result is None


class TestGenerateCachedStructureImage:
    """Test cached structure image generation."""

    @patch("app.similarity_search.visualization.get_molecule")
    def test_invalid_query_molecule(self, mock_get_mol):
        """Test when query molecule cannot be parsed - covers line 143."""
        _generate_cached_structure_image.cache_clear()
        mock_get_mol.return_value = None  # Invalid molecule

        result = _generate_cached_structure_image("q1", "INVALID", "r1", "CC")

        assert result is None

    @patch("app.similarity_search.visualization.get_molecule")
    def test_invalid_reference_molecule(self, mock_get_mol):
        """Test when reference molecule cannot be parsed - covers line 143."""
        _generate_cached_structure_image.cache_clear()
        mock_query_mol = MagicMock()
        mock_get_mol.side_effect = [mock_query_mol, None]  # Reference fails

        result = _generate_cached_structure_image("q1", "CCO", "r1", "INVALID")

        assert result is None

    @patch("app.similarity_search.visualization.get_molecule")
    def test_exception_during_image_generation(self, mock_get_mol):
        """Test exception handling during image generation - covers lines 153 and 155-157."""
        _generate_cached_structure_image.cache_clear()
        mock_query_mol = MagicMock()
        mock_ref_mol = MagicMock()
        mock_get_mol.side_effect = [mock_query_mol, mock_ref_mol]

        # Patch Draw to raise exception
        with patch(
            "app.similarity_search.visualization.Draw.MolToImage",
            side_effect=Exception("Draw failed"),
        ):
            result = _generate_cached_structure_image("q1", "CCO", "r1", "CC")

        assert result is None


class TestVisualizeDistributionEdgeCases:
    """Test ranking plot visualization edge cases."""

    @patch("app.similarity_search.visualization.st")
    def test_exception_in_visualization(self, mock_st):
        """Test exception handling in visualize_distribution - covers except block."""
        # Set up st.write to raise an exception
        mock_st.write.side_effect = Exception("Test error")

        top_hits = pd.DataFrame(
            {
                "query_name": ["Query1"],
                "ref_name": ["Ref1"],
                "similarity": [0.9],
            }
        )

        query_df = pd.DataFrame(
            {
                "query_name": ["Query1"],
            }
        )

        result = visualize_distribution(top_hits, query_df, show=True)

        # Should catch exception and return empty dict
        assert result == {}
        # Should have called st.error
        mock_st.error.assert_called_once()


class TestPrepareCSVExport:
    """Test CSV export preparation."""

    def test_csv_export_columns_dropped_correctly(self):
        """Test that Structures column is properly dropped - covers line 177."""
        results_df = pd.DataFrame(
            {
                "Query Molecule": ["Q1"],
                "Query SMILES": ["CCO"],
                "Reference Molecule": ["R1"],
                "Reference SMILES": ["CC"],
                "Similarity Score": [0.9],
                "Structures": ["<img>"],
            }
        )

        csv_str = prepare_csv_export(results_df)

        # Verify Structures is not in output
        assert "Structures" not in csv_str
        # Verify required columns are present
        assert "Query Molecule" in csv_str
        assert "Q1" in csv_str


class TestVisualizeDistribution:
    """Test ranking plot visualization."""

    @patch("app.similarity_search.visualization.st")
    def test_exception_in_visualization(self, mock_st):
        """Test exception handling in visualize_distribution - covers except block."""
        # Set up st.write to raise an exception
        mock_st.write.side_effect = Exception("Test error")

        top_hits = pd.DataFrame(
            {
                "query_name": ["Query1"],
                "ref_name": ["Ref1"],
                "similarity": [0.9],
            }
        )

        query_df = pd.DataFrame(
            {
                "query_name": ["Query1"],
            }
        )

        result = visualize_distribution(top_hits, query_df, show=True)

        # Should catch exception and return empty dict
        assert result == {}
        # Should have called st.error
        mock_st.error.assert_called_once()
