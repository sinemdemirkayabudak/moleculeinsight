"""
Molecular Similarity Search Package

Using Morgan fingerprints + Tanimoto similarity metric for fast, efficient
molecular screening against large chemical libraries.
"""

from app.similarity_search.cli import create_argument_parser, main, run_similarity_search
from app.similarity_search.fingerprints import get_morgan_fp, similarity_search
from app.similarity_search.pipeline import (
    compute_fingerprints,
    convert_smiles_to_molecules,
    rank_results,
    search_similarities,
)
from app.similarity_search.validators import validate_dataframe, validate_parameters
from app.similarity_search.visualization import (
    create_structure_image,
    prepare_csv_export,
    visualize_distribution,
)

__all__ = [
    # Fingerprinting
    "get_morgan_fp",
    "similarity_search",
    # Validation
    "validate_parameters",
    "validate_dataframe",
    # Pipeline
    "convert_smiles_to_molecules",
    "compute_fingerprints",
    "search_similarities",
    "rank_results",
    # Visualization
    "visualize_distribution",
    "create_structure_image",
    "prepare_csv_export",
    # CLI
    "run_similarity_search",
    "create_argument_parser",
    "main",
]
