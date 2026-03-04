"""
Molecular Similarity Search Package

Using Morgan fingerprints + Tanimoto similarity metric for fast, efficient
molecular screening against large chemical libraries.
"""

# Export main public API functions for backward compatibility
from app.similarity_search.fingerprints import get_morgan_fp, similarity_search
from app.similarity_search.validators import validate_parameters, validate_dataframe
from app.similarity_search.pipeline import (
    convert_smiles_to_molecules,
    compute_fingerprints,
    search_similarities,
    rank_results
)
from app.similarity_search.visualization import (
    visualize_distribution,
    create_structure_image,
    prepare_csv_export
)
from app.similarity_search.cli import (
    run_similarity_search,
    create_argument_parser,
    main
)

# Re-export dependencies for test mocking compatibility
import pandas as pd
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import TanimotoSimilarity
from PIL import Image
import matplotlib.pyplot as plt

__all__ = [
    # Fingerprinting
    'get_morgan_fp',
    'similarity_search',
    # Validation
    'validate_parameters',
    'validate_dataframe',
    # Pipeline
    'convert_smiles_to_molecules',
    'compute_fingerprints',
    'search_similarities',
    'rank_results',
    # Visualization
    'visualize_distribution',
    'create_structure_image',
    'prepare_csv_export',
    # CLI
    'run_similarity_search',
    'create_argument_parser',
    'main',
]
