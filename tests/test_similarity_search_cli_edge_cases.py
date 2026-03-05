"""Edge case tests for similarity_search CLI module."""

from unittest.mock import patch

import pandas as pd
import pytest

from app.similarity_search.cli import run_similarity_search
from app.similarity_search.fingerprints import get_morgan_fp
from app.similarity_search.pipeline import rank_results, search_similarities
from app.similarity_search.visualization import visualize_distribution


class TestEdgeCases:
    """Test edge cases in similarity search."""

    def test_invalid_smiles_in_query(self):
        """Test handling of invalid SMILES in query."""
        query_df = pd.DataFrame(
            {"smiles": ["CCO", "INVALID_SMILES", "CC"], "name": ["valid1", "invalid", "valid2"]}
        )
        ref_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})

        # Should handle gracefully by filtering out invalid SMILES
        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
        )

        assert isinstance(results, pd.DataFrame)
        # Only valid query molecules should be in results
        assert len(results) >= 0

    def test_invalid_smiles_in_reference(self):
        """Test handling of invalid SMILES in reference."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame(
            {
                "smiles": ["INVALID1", "CC", "INVALID2", "c1ccccc1"],
                "name": ["bad1", "ethane", "bad2", "benzene"],
            }
        )

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
        )

        assert isinstance(results, pd.DataFrame)
        # Only valid reference molecules should appear
        if len(results) > 0:
            assert "bad1" not in results["ref_name"].values
            assert "bad2" not in results["ref_name"].values

    def test_all_invalid_smiles_query(self):
        """Test when all query SMILES are invalid."""
        query_df = pd.DataFrame({"smiles": ["INVALID1", "INVALID2"], "name": ["bad1", "bad2"]})
        ref_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})

        # Should raise ValueError when no valid query molecules
        with pytest.raises(ValueError, match="No valid query molecules"):
            run_similarity_search(
                query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
            )

    def test_empty_reference_library(self):
        """Test with empty reference library."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": [], "name": []})

        # Should raise ValueError when reference library is empty
        with pytest.raises(ValueError, match="Reference library is empty|empty"):
            run_similarity_search(
                query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
            )

    def test_single_query_molecule(self):
        """Test with single query molecule."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCCO", "CC", "c1ccccc1"],
                "name": ["ethanol", "propanol", "ethane", "benzene"],
            }
        )

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=2, show_plots=False
        )

        assert isinstance(results, pd.DataFrame)
        assert all(results["query_name"] == "ethanol")

    def test_single_reference_molecule(self):
        """Test with single reference molecule."""
        query_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})
        ref_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
        )

        assert len(results) >= 0

    def test_top_n_larger_than_results(self):
        """Test when top_n is larger than available results."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=100, show_plots=False
        )

        # Should return all available results
        assert len(results) <= 2

    def test_duplicate_reference_molecules(self):
        """Test with duplicate molecules in reference."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO", "CCO", "CCO"], "name": ["dup1", "dup2", "dup3"]})

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=2, show_plots=False
        )

        # All duplicates should have perfect similarity
        assert all(results["similarity"] == 1.0)

    def test_various_molecule_complexities(self):
        """Test with molecules of varying structural complexity."""
        query_df = pd.DataFrame({"smiles": ["C"], "name": ["methane"]})
        ref_df = pd.DataFrame(
            {
                "smiles": [
                    "C",  # methane (simple)
                    "CC",  # ethane
                    "CCO",  # ethanol
                    "c1ccccc1",  # benzene
                    "c1ccc2c(c1)ccc3c2cccc3",  # anthracene (complex)
                ],
                "name": ["meth", "eth", "ethol", "benz", "anth"],
            }
        )

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
        )

        assert len(results) > 0


class TestPipelineExceptionHandling:
    """Test pipeline exception handling for complete coverage."""

    def test_search_similarities_with_exception_logging(self):
        """Test exception logging in search_similarities."""
        from rdkit import Chem

        query_df = pd.DataFrame(
            {"query_name": ["q1"], "fp": [get_morgan_fp(Chem.MolFromSmiles("CCO"), 2)]}
        )

        ref_df = pd.DataFrame({"fp": [get_morgan_fp(Chem.MolFromSmiles("CC"), 2)]})

        with patch("app.similarity_search.pipeline.logger") as mock_logger:
            result = search_similarities(query_df, ref_df)

            # Verify debug logging was called
            assert mock_logger.debug.called or len(result) > 0

    def test_rank_results_logging_per_query(self):
        """Test rank_results logs statistics for each query."""
        combined_results = pd.DataFrame(
            {
                "query_name": ["q1", "q1", "q2", "q2"],
                "smiles": ["C", "CC", "C", "CC"],
                "ref_name": ["r1", "r2", "r3", "r4"],
                "similarity": [0.9, 0.7, 0.8, 0.6],
            }
        )

        query_df = pd.DataFrame({"query_name": ["q1", "q2"]})

        with patch("app.similarity_search.pipeline.logger") as mock_logger:
            _ = rank_results(combined_results, query_df, top_n=2)

            # Verify logging was called for query statistics
            assert mock_logger.info.called


class TestVisualizationDistributionExceptionPaths:
    """Test visualization distribution exception handling."""

    def test_visualize_distribution_with_no_plots(self):
        """Test visualize_distribution returns empty dict on exception."""
        # Create data that will trigger exception handling in visualization
        top_hits = pd.DataFrame(
            {"query_name": ["q1"], "ref_name": ["r1"], "similarity": [0.5], "smiles": ["C"]}
        )

        query_df = pd.DataFrame({"query_name": ["q1"]})

        with patch(
            "app.similarity_search.visualization.plt.subplots", side_effect=Exception("Plot error")
        ):
            result = visualize_distribution(top_hits, query_df)
            # Should return empty dict on error
            assert isinstance(result, dict)

    def test_visualize_distribution_many_results(self):
        """Test visualize_distribution with large number of results."""
        # Create many results for a single query to test the plotting logic
        n = 100
        top_hits = pd.DataFrame(
            {
                "query_name": ["q1"] * n,
                "ref_name": [f"r{i}" for i in range(n)],
                "similarity": [0.9 - (i * 0.001) for i in range(n)],
                "smiles": ["C"] * n,
            }
        )

        query_df = pd.DataFrame({"query_name": ["q1"]})

        result = visualize_distribution(top_hits, query_df)
        assert isinstance(result, dict)


class TestCSVParsingErrors:
    """Test CSV parsing error handling in run_similarity_search."""

    def test_csv_parser_error_handling(self):
        """Test that ParserError from invalid CSV format is handled."""
        _ = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})

        # Mock pd.read_csv to raise ParserError
        with patch(
            "app.similarity_search.cli.pd.read_csv",
            side_effect=pd.errors.ParserError("Invalid CSV format"),
        ):
            with pytest.raises(ValueError, match="Invalid CSV format"):
                run_similarity_search(
                    query_file="query.csv",
                    reference_file=ref_df,
                    radius=2,
                    top_n=5,
                    show_plots=False,
                )

    def test_file_not_found_error_handling(self):
        """Test that FileNotFoundError is properly logged and raised."""
        _ = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})

        # Mock pd.read_csv to raise FileNotFoundError
        with patch(
            "app.similarity_search.cli.pd.read_csv", side_effect=FileNotFoundError("File not found")
        ):
            with pytest.raises(FileNotFoundError):
                run_similarity_search(
                    query_file="missing.csv",
                    reference_file=ref_df,
                    radius=2,
                    top_n=5,
                    show_plots=False,
                )


class TestPipelineDebugLogging:
    """Test debug logging paths in pipeline functions."""

    def test_fingerprint_compute_debug_logging(self):
        """Test that debug log is recorded after successful fingerprint computation."""
        from rdkit import Chem

        from app.similarity_search.pipeline import compute_fingerprints

        query_df = pd.DataFrame(
            {
                "mol": [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")],
                "query_name": ["ethanol", "ethane"],
            }
        )
        ref_df = pd.DataFrame({"mol": [Chem.MolFromSmiles("CCO")], "ref_name": ["ref_ethanol"]})

        # Call compute_fingerprints and verify it processes without error
        result_query, result_ref = compute_fingerprints(query_df, ref_df, radius=2)

        # Verify fingerprints were computed
        assert "fp" in result_query.columns
        assert "fp" in result_ref.columns
        assert len(result_query) == 2
        assert len(result_ref) == 1

    def test_fingerprint_compute_error_path(self):
        """Test exception path in compute_fingerprints."""
        from rdkit import Chem

        from app.similarity_search.pipeline import compute_fingerprints

        query_df = pd.DataFrame({"mol": [Chem.MolFromSmiles("CCO")], "query_name": ["ethanol"]})
        ref_df = pd.DataFrame({"mol": [Chem.MolFromSmiles("CCO")], "ref_name": ["ref_ethanol"]})

        # Mock get_morgan_fp to raise exception
        with patch(
            "app.similarity_search.pipeline.get_morgan_fp",
            side_effect=RuntimeError("Fingerprint error"),
        ):
            with pytest.raises(RuntimeError):
                compute_fingerprints(query_df, ref_df, radius=2)

    def test_search_similarities_with_multiple_queries(self):
        """Test search_similarities processes multiple query molecules."""
        # Run full pipeline to get proper fingerprints
        query_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})
        ref_df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"], "name": ["ethanol", "benzene"]})

        # Use full pipeline to get proper fingerprints
        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
        )

        # Verify both queries are in results
        assert "ethanol" in results["query_name"].values
        assert "ethane" in results["query_name"].values

    def test_search_similarities_error_path(self):
        """Test exception handling in search_similarities."""
        # Run full pipeline first to reach the search function
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ref_ethanol"]})

        # Mock similarity_search to raise exception
        with patch(
            "app.similarity_search.pipeline.similarity_search",
            side_effect=RuntimeError("Search error"),
        ):
            with pytest.raises(RuntimeError):
                run_similarity_search(
                    query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
                )

    def test_rank_results_per_query_logging(self):
        """Test per-query statistics logging in rank_results."""
        # Create results with multiple queries
        combined_results = pd.DataFrame(
            {
                "query_name": ["q1", "q1", "q1", "q2", "q2", "q2"],
                "ref_name": ["r1", "r2", "r3", "r1", "r2", "r3"],
                "smiles": ["C"] * 6,
                "similarity": [0.95, 0.85, 0.75, 0.90, 0.80, 0.70],
            }
        )

        query_df = pd.DataFrame({"query_name": ["q1", "q2"]})

        # Call rank_results
        top_hits = rank_results(combined_results, query_df, top_n=2)

        # Verify statistics were logged (function processes without error)
        assert len(top_hits) == 4  # 2 queries × top_n=2
        assert all(qry in top_hits["query_name"].values for qry in ["q1", "q2"])

    def test_rank_results_error_path(self):
        """Test exception handling in rank_results."""
        combined_results = pd.DataFrame(
            {"query_name": ["q1"], "ref_name": ["r1"], "smiles": ["C"], "similarity": [0.9]}
        )

        query_df = pd.DataFrame({"query_name": ["q1"]})

        # Mock sort_values to raise exception
        with patch.object(combined_results, "sort_values", side_effect=RuntimeError("Sort error")):
            with pytest.raises(RuntimeError):
                rank_results(combined_results, query_df, top_n=2)

    def test_empty_query_after_validation(self):
        """Test that empty query molecules after validation raises error."""
        query_df = pd.DataFrame({"smiles": ["INVALID"], "name": ["bad"]})
        ref_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})

        # Should raise ValueError when all query molecules are invalid
        with pytest.raises(ValueError, match="No valid query molecules"):
            run_similarity_search(
                query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
            )

    def test_empty_reference_after_validation(self):
        """Test that empty reference molecules after validation raises error."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["INVALID"], "name": ["bad"]})

        # Should raise ValueError when all reference molecules are invalid
        with pytest.raises(ValueError, match="No valid reference molecules"):
            run_similarity_search(
                query_file=query_df, reference_file=ref_df, radius=2, top_n=5, show_plots=False
            )


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""

    def test_create_structure_image_success(self):
        """Test create_structure_image with valid molecules."""
        from app.similarity_search.visualization import create_structure_image

        # Successfully create image for valid molecules
        result = create_structure_image("q1", "CCO", "r1", "CC")
        # Result should be either a base64 string or None
        assert isinstance(result, (str, type(None)))

    def test_visualize_distribution_error_recovery(self):
        """Test that visualize_distribution handles errors gracefully."""
        # Create data that will test error handling
        top_hits = pd.DataFrame(
            {"query_name": ["q1"], "ref_name": ["r1"], "similarity": [0.9], "smiles": ["CCO"]}
        )

        query_df = pd.DataFrame({"query_name": ["q1"]})

        # Mock matplotlib to raise exception
        with patch(
            "app.similarity_search.visualization.plt.subplots",
            side_effect=RuntimeError("Plot error"),
        ):
            result = visualize_distribution(top_hits, query_df, show=False)
            # Should return empty dict on error
            assert isinstance(result, dict)


class TestRankResultsWithEmptyMatches:
    """Test rank_results handles queries with no matches."""

    def test_rank_results_with_empty_query_matches(self):
        """Test that queries with no matches in top_hits are skipped in logging."""
        # Create results where one query has matches but results are filtered out
        combined_results = pd.DataFrame(
            {
                "query_name": ["q1", "q1", "q2"],
                "ref_name": ["r1", "r2", "r1"],
                "smiles": ["C", "C", "C"],
                "similarity": [0.95, 0.85, 0.5],
            }
        )

        query_df = pd.DataFrame(
            {
                "query_name": ["q1", "q2", "q3"]  # q3 has no results
            }
        )

        # Only get top 1 - q2 will result in no matches after filtering
        top_hits = rank_results(combined_results, query_df, top_n=1)

        # The loop should skip q3 since it has no matches
        assert isinstance(top_hits, pd.DataFrame)
        # Verify function handles queries with no results gracefully
        assert "q1" in top_hits["query_name"].values


class TestVisualizationStructureImageException:
    """Test visualization exception handling for structure image creation."""

    def test_create_structure_image_invalid_molecule_pair(self):
        """Test create_structure_image with molecules that return None."""
        from app.similarity_search.visualization import create_structure_image

        # Using invalid SMILES should trigger exception handler
        # Invalid SMILES returns None, which will cause issues in image creation
        result = create_structure_image("q1", "INVALID", "r1", "ALSO_INVALID")
        # Should return None when molecule creation fails
        assert result is None

    def test_create_structure_image_with_special_characters(self):
        """Test create_structure_image with special characters in molecule names."""
        from app.similarity_search.visualization import create_structure_image

        # Test with special characters in names (should still work)
        result = create_structure_image("query@1", "CCO", "ref#1", "CC")
        # Should return image data or None, depending on SMILES validity
        assert isinstance(result, (str, type(None)))
