# ============================================================
# COMMAND LINE INTERFACE & ORCHESTRATION
# ============================================================

import argparse
import logging
import warnings
import pandas as pd

from app.similarity_search.validators import validate_parameters, validate_dataframe
from app.similarity_search.pipeline import (
    convert_smiles_to_molecules,
    compute_fingerprints,
    search_similarities,
    rank_results
)
from app.similarity_search.visualization import visualize_distribution

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent propagation to parent loggers (stops duplicate logs)

# Remove existing handlers to avoid duplicates
# prevents duplicate handlers when code reruns
logger.handlers.clear()

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def run_similarity_search(
        query_file,
        reference_file,
        radius,
        top_n,
        show_plots
):
    """Run the complete Morgan fingerprint similarity search pipeline.
    
    Parameters:
        query_file (str or pd.DataFrame): Path to CSV file OR DataFrame containing query molecules
                                         Must have 'name' and 'smiles' columns
        reference_file (str or pd.DataFrame): Path to CSV file OR DataFrame containing reference library
                                             Must have 'name' and 'smiles' columns
        radius (int): Morgan fingerprint radius (0-5, typically 2 for ECFP4)
        top_n (int): Number of top similar molecules to return per query
        show_plots (bool): Whether to generate ranking plots
    
    Returns:
        tuple: (figures dict, top_hits dataframe) where figures maps query_name -> matplotlib figure object
    
    Raises:
        FileNotFoundError: If input files don't exist (when str paths provided)
        ValueError: If parameters are invalid or dataframes incomplete/empty
        Exception: For other unexpected errors during pipeline execution
        
    Note:
        Accepts DataFrames to enable in-memory processing, eliminating disk I/O for streaming/uploaded files.
        Backwards compatible with file path strings (existing code works unchanged).
    """
    try:
        # Validate parameters
        validate_parameters(radius, top_n)
        
        logger.info("Loading datasets...")

        # Load query molecules and reference library (accept DataFrames or file paths)
        try:
            if isinstance(query_file, pd.DataFrame):
                query_df = query_file.copy()  # Copy to avoid mutating input
            else:
                query_df = pd.read_csv(query_file)
            
            if isinstance(reference_file, pd.DataFrame):
                ref_df = reference_file.copy()  # Copy to avoid mutating input
            else:
                ref_df = pd.read_csv(reference_file)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise ValueError(f"Invalid CSV format: {e}")
        
        logger.info(f"Query dataset: {len(query_df)} molecules")
        logger.info(f"Reference library: {len(ref_df)} molecules")
        
        # Validate dataframes
        validate_dataframe(query_df, "Query dataset", ["smiles", "name"])
        validate_dataframe(ref_df, "Reference library", ["smiles", "name"])
        
        # Rename molecules' name column for output clarity
        query_df = query_df.rename(columns={"name": "query_name"})
        ref_df = ref_df.rename(columns={"name": "ref_name"})

        # Convert SMILES to molecules
        query_df, ref_df = convert_smiles_to_molecules(query_df, ref_df)
        
        # Compute fingerprints
        query_df, ref_df = compute_fingerprints(query_df, ref_df, radius)
        
        # Search similarities
        combined_results = search_similarities(query_df, ref_df)
        
        # Rank results (in-memory, no file I/O)
        top_hits = rank_results(combined_results, query_df, top_n)
        
        # Collect figures for display
        figures = {}
        
        # Pre-generate all plots upfront for instant dropdown selection
        if show_plots:
            figures = visualize_distribution(top_hits, query_df, show=False)
        
        logger.info("Similarity search pipeline completed successfully")
        
        return figures, top_hits
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def create_argument_parser():
    """
    Create and configure the command line argument parser.
    
    Returns:
        argparse.ArgumentParser configured with all options
    """
    parser = argparse.ArgumentParser(
        description="Morgan Fingerprint Similarity Search Pipeline"
    )

    # Input files
    parser.add_argument("--query_file", required=True,
                        help="CSV file containing query molecules")

    parser.add_argument("--reference_file", required=True,
                        help="CSV file containing reference library")

    # Output
    parser.add_argument("--output_file",
                        default="similarity_results.csv",
                        help="Output results file")

    # Fingerprint parameters
    parser.add_argument("--radius",
                        type=int,
                        default=2,
                        help="Morgan fingerprint radius")

    # Ranking
    parser.add_argument("--top_n",
                        type=int,
                        default=20,
                        help="Top N similar molecules")

    # Visualization options
    parser.add_argument("--show_plots",
                        action="store_true",
                        help="Generate similarity rank plots")

    return parser


# -----------------------------
# Command Line Interface
# -----------------------------

def main():
    """Main entry point with error handling."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()

        # Run pipeline (for CLI, show plots = True to display immediately)
        run_similarity_search(
            query_file=args.query_file,
            reference_file=args.reference_file,
            output_file=args.output_file,
            radius=args.radius,
            top_n=args.top_n,
            show_plots=args.show_plots
        )
    
    except ValueError as e:
        logger.error(f"Invalid parameter: {e}")
        exit(1)
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


# Entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        exit(0)
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        exit(1)
