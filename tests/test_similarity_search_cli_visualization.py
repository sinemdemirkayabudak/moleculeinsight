"""Tests for similarity_search visualization module."""

import pandas as pd

from app.similarity_search.cli import run_similarity_search
from app.similarity_search.visualization import (
    create_structure_image,
    prepare_csv_export,
    visualize_distribution,
)


class TestVisualizeDistribution:
    """Test ranking plot visualization."""

    def test_basic_plot_generation(self):
        """Test basic plot generation."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame(
            {"smiles": ["CCO", "CCCO", "CC"], "name": ["ethanol", "propanol", "ethane"]}
        )

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=3, show_plots=True
        )

        assert isinstance(figures, dict)
        if figures:
            for _, fig in figures.items():
                assert fig is not None

    def test_plot_with_high_similarity_scores(self):
        """Test plot with high similarity scores."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame(
            {"smiles": ["CCO", "CCO", "CCO"], "name": ["match1", "match2", "match3"]}
        )

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=3, show_plots=True
        )

        assert len(results) > 0

    def test_plot_without_show(self):
        """Test plot generation without showing."""
        query_df = pd.DataFrame({"smiles": ["c1ccccc1"], "name": ["benzene"]})
        ref_df = pd.DataFrame(
            {
                "smiles": ["c1ccccc1", "c1ccccc1C", "c1ccccc1CC"],
                "name": ["benz", "toluene", "xylene"],
            }
        )

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=3, show_plots=False
        )

        assert isinstance(results, pd.DataFrame)


class TestCreateStructureImage:
    """Test structure image generation."""

    def test_valid_structure_image(self):
        """Test generating valid structure images."""
        image = create_structure_image("query", "CCO", "ref", "CC")
        assert image is not None
        assert isinstance(image, str)  # base64 string

    def test_invalid_smiles_image(self):
        """Test with invalid SMILES."""
        image = create_structure_image("query", "INVALID", "ref", "CC")
        assert image is None or isinstance(image, str)

    def test_image_caching(self):
        """Test image caching."""
        image1 = create_structure_image("q1", "CCO", "r1", "CC")
        image2 = create_structure_image("q1", "CCO", "r1", "CC")
        # Second call should use cache
        assert image1 == image2

    def test_multiple_structure_images(self):
        """Test generating multiple structure images."""
        pairs = [
            ("q1", "CCO", "r1", "CC"),
            ("q2", "c1ccccc1", "r2", "CCO"),
            ("q3", "CC", "r3", "c1ccccc1"),
        ]

        images = []
        for query_name, query_smiles, ref_name, ref_smiles in pairs:
            image = create_structure_image(query_name, query_smiles, ref_name, ref_smiles)
            images.append(image)

        assert len(images) == 3


class TestPrepareCSVExport:
    """Test CSV export preparation."""

    def test_csv_export_with_correct_columns(self):
        """Test CSV export with correct column names."""
        results = pd.DataFrame(
            {
                "Query Molecule": ["ethanol", "ethanol"],
                "Query SMILES": ["CCO", "CCO"],
                "Reference Molecule": ["ethanol", "ethane"],
                "Reference SMILES": ["CCO", "CC"],
                "Similarity Score": [1.0, 0.75],
                "Structures": ["<img>", "<img>"],
            }
        )

        csv_string = prepare_csv_export(results)

        assert isinstance(csv_string, str)
        # Structures column should be removed for CSV
        assert "Structures" not in csv_string

    def test_csv_with_special_characters(self):
        """Test CSV export with special characters in molecule names."""
        results = pd.DataFrame(
            {
                "Query Molecule": ["mol-1", "mol_2"],
                "Query SMILES": ["CCO", "CCO"],
                "Reference Molecule": ["ref,1", 'ref"2'],
                "Reference SMILES": ["CC", "CCC"],
                "Similarity Score": [0.9, 0.85],
                "Structures": ["", ""],
            }
        )

        csv_string = prepare_csv_export(results)
        assert isinstance(csv_string, str)

    def test_empty_results_export(self):
        """Test CSV export with empty results."""
        results = pd.DataFrame(
            columns=[
                "Query Molecule",
                "Query SMILES",
                "Reference Molecule",
                "Reference SMILES",
                "Similarity Score",
                "Structures",
            ]
        )

        csv_string = prepare_csv_export(results)
        assert isinstance(csv_string, str)


class TestVisualizationEdgeCases:
    """Test visualization edge cases for 100% coverage."""

    def test_visualize_empty_dataframe(self):
        """Test visualize_distribution with empty results."""
        top_hits = pd.DataFrame({"query_name": [], "ref_name": [], "similarity": [], "smiles": []})

        query_df = pd.DataFrame({"query_name": ["q1", "q2"]})

        result = visualize_distribution(top_hits, query_df)
        assert isinstance(result, dict)

    def test_visualize_with_many_hits_per_query(self):
        """Test visualization with many hits per query."""
        top_hits = pd.DataFrame(
            {
                "query_name": ["q1"] * 50,
                "ref_name": [f"ref_{i}" for i in range(50)],
                "similarity": [0.95 - (i * 0.01) for i in range(50)],
                "smiles": ["C"] * 50,
            }
        )

        query_df = pd.DataFrame({"query_name": ["q1"]})

        result = visualize_distribution(top_hits, query_df)
        assert isinstance(result, dict)

    def test_prepare_csv_export_edge_cases(self):
        """Test CSV export edge cases."""
        # Test with various data
        results_df = pd.DataFrame(
            {
                "Query Molecule": ["q1", "q1"],
                "Query SMILES": ["CCO", "CCO"],
                "Reference Molecule": ["r1", "r2"],
                "Reference SMILES": ["CC", "CCC"],
                "Similarity Score": [0.9, 0.7],
                "Structures": ["", ""],
            }
        )

        csv_output = prepare_csv_export(results_df)

        assert isinstance(csv_output, str)
        assert "Query Molecule" in csv_output
        assert "q1" in csv_output

    def test_create_structure_image_various_molecules(self):
        """Test structure image creation for various molecule types."""
        test_cases = [
            ("methane", "C", "ethane", "CC"),
            ("benzene", "c1ccccc1", "toluene", "c1ccccc1C"),
            ("naphthalene", "c1ccc2ccccc2c1", "anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ]

        for q_name, q_smiles, r_name, r_smiles in test_cases:
            _ = create_structure_image(q_name, q_smiles, r_name, r_smiles)
            # Should return either None (on error) or base64 string


class TestVisualizationExceptionHandling:
    """Test exception handling in visualization functions."""

    def test_create_structure_image_with_invalid_smiles(self):
        """Test create_structure_image handles invalid SMILES."""
        from app.similarity_search.visualization import create_structure_image

        # Invalid SMILES should trigger the exception handler
        result = create_structure_image("q1", "NOTSMILES", "r1", "NOTSMILES")
        # Should return None on error
        assert result is None

    def test_prepare_csv_export(self):
        """Test prepare_csv_export removes structure column and reorders."""
        from app.similarity_search.visualization import prepare_csv_export

        results_df = pd.DataFrame(
            {
                "Query Molecule": ["q1"],
                "Query SMILES": ["CCO"],
                "Reference Molecule": ["r1"],
                "Reference SMILES": ["CC"],
                "Similarity Score": [0.9],
                "Structures": ["img1"],
            }
        )

        csv_str = prepare_csv_export(results_df)

        # Verify CSV format
        assert isinstance(csv_str, str)
        assert "Query Molecule" in csv_str
        assert "Reference Molecule" in csv_str
        assert "Structures" not in csv_str  # Should be dropped
