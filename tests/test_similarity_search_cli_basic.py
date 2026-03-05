"""Basic tests for similarity_search CLI module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.similarity_search.cli import main, run_similarity_search


class TestRunSimilaritySearch:
    """Test main CLI function."""

    def test_basic_search_with_files(self):
        """Test basic similarity search with CSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            query_file = Path(tmp_dir) / "query.csv"
            ref_file = Path(tmp_dir) / "reference.csv"

            query_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})
            ref_df = pd.DataFrame(
                {"smiles": ["CCO", "CCCO", "c1ccccc1"], "name": ["ethanol", "propanol", "benzene"]}
            )

            query_df.to_csv(query_file, index=False)
            ref_df.to_csv(ref_file, index=False)

            figures, results = run_similarity_search(
                query_file=str(query_file),
                reference_file=str(ref_file),
                radius=2,
                top_n=2,
                show_plots=False,
            )

            assert isinstance(results, pd.DataFrame)
            assert "similarity" in results.columns
            assert len(results) > 0

    def test_search_with_dataframes(self):
        """Test similarity search with pandas DataFrames."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=1, show_plots=False
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1

    def test_custom_radius(self):
        """Test with custom fingerprint radius."""
        query_df = pd.DataFrame({"smiles": ["c1ccccc1"], "name": ["benzene"]})
        ref_df = pd.DataFrame({"smiles": ["c1ccccc1", "c1ccccc1C"], "name": ["benzene", "toluene"]})

        for radius in [0, 2, 5]:
            figures, results = run_similarity_search(
                query_file=query_df, reference_file=ref_df, radius=radius, top_n=2, show_plots=False
            )
            assert isinstance(results, pd.DataFrame)

    def test_default_parameters(self):
        """Test with default parameters."""
        query_df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        ref_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})

        figures, results = run_similarity_search(
            query_file=query_df, reference_file=ref_df, radius=2, top_n=20, show_plots=False
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 20  # top_n=20

    def test_main_function_with_valid_files(self):
        """Test main() CLI entry point with valid files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            query_file = Path(tmp_dir) / "query.csv"
            ref_file = Path(tmp_dir) / "reference.csv"

            query_df = pd.DataFrame({"smiles": ["CCO", "CC"], "name": ["ethanol", "ethane"]})
            ref_df = pd.DataFrame({"smiles": ["CCO", "CCCO"], "name": ["ethanol", "propanol"]})

            query_df.to_csv(query_file, index=False)
            ref_df.to_csv(ref_file, index=False)

            test_args = [
                "script.py",
                "--query_file",
                str(query_file),
                "--reference_file",
                str(ref_file),
                "--radius",
                "2",
                "--top_n",
                "10",
            ]

            with patch.object(sys, "argv", test_args):
                with patch("app.similarity_search.cli.run_similarity_search") as mock_run:
                    mock_run.return_value = ({}, pd.DataFrame())
                    main()
                    assert mock_run.called

    def test_main_function_value_error(self):
        """Test main() handles ValueError from run_similarity_search."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            query_file = Path(tmp_dir) / "empty_query.csv"
            ref_file = Path(tmp_dir) / "ref.csv"

            # Create empty query file
            query_file.write_text("smiles,name")
            ref_file.write_text("smiles,name\nCC,ethane")

            test_args = [
                "script.py",
                "--query_file",
                str(query_file),
                "--reference_file",
                str(ref_file),
            ]

            with patch.object(sys, "argv", test_args):
                with patch("app.similarity_search.cli.exit") as mock_exit:
                    main()
                    # Verify exit(1) was called on ValueError
                    mock_exit.assert_called_with(1)

    def test_main_function_file_not_found(self):
        """Test main() handles FileNotFoundError."""
        test_args = [
            "script.py",
            "--query_file",
            "/nonexistent/query.csv",
            "--reference_file",
            "/nonexistent/ref.csv",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("app.similarity_search.cli.exit") as mock_exit:
                main()
                mock_exit.assert_called_with(1)

    def test_main_function_generic_exception(self):
        """Test main() handles generic Exception."""
        test_args = ["script.py", "--query_file", "query.csv", "--reference_file", "ref.csv"]

        with patch.object(sys, "argv", test_args):
            with patch(
                "app.similarity_search.cli.run_similarity_search",
                side_effect=RuntimeError("Unexpected error"),
            ):
                with patch("app.similarity_search.cli.exit") as mock_exit:
                    main()
                    # Verify exit(1) was called on generic Exception
                    mock_exit.assert_called_with(1)

    def test_main_function_value_error_exit(self):
        """Test main() exit(1) is called on ValueError."""
        test_args = ["script.py", "--query_file", "query.csv", "--reference_file", "ref.csv"]

        with patch.object(sys, "argv", test_args):
            with patch(
                "app.similarity_search.cli.run_similarity_search",
                side_effect=ValueError("Invalid parameter"),
            ):
                with patch("app.similarity_search.cli.exit") as mock_exit:
                    main()
                    # Verify exit(1) was called on ValueError
                    mock_exit.assert_called_with(1)

    def test_main_function_file_not_found_exit(self):
        """Test main() exit(1) is called on FileNotFoundError."""
        test_args = [
            "script.py",
            "--query_file",
            "/missing/query.csv",
            "--reference_file",
            "/missing/ref.csv",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "app.similarity_search.cli.run_similarity_search",
                side_effect=FileNotFoundError("File not found"),
            ):
                with patch("app.similarity_search.cli.exit") as mock_exit:
                    main()
                    # Verify exit(1) was called on FileNotFoundError
                    mock_exit.assert_called_with(1)
                    mock_exit.assert_called_with(1)
