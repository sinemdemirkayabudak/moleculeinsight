# ============================================================
# SIMILARITY SEARCH PIPELINE
# ============================================================

import logging
import warnings
import pandas as pd

from app.molecule import get_molecule
from app.similarity_search.fingerprints import get_morgan_fp, similarity_search

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


def convert_smiles_to_molecules(query_df, ref_df):
    """
    Convert SMILES strings to RDKit molecule objects.
    Removes invalid molecules.
    
    Returns:
        Tuple of (cleaned query_df, cleaned ref_df)
    """
    logger.info("Converting SMILES strings to molecular objects...")

    query_df["mol"] = query_df["smiles"].apply(get_molecule)
    ref_df["mol"] = ref_df["smiles"].apply(get_molecule)
    query_before = len(query_df)
    ref_before = len(ref_df)
    query_df = query_df.dropna(subset=["mol"])
    ref_df = ref_df.dropna(subset=["mol"])
    logger.info(f"Valid query molecules: {len(query_df)}/{query_before}")
    logger.info(f"Valid reference molecules: {len(ref_df)}/{ref_before}")
    
    if len(query_df) == 0:
        raise ValueError("No valid query molecules after SMILES validation")
    if len(ref_df) == 0:
        raise ValueError("No valid reference molecules after SMILES validation")
    
    return query_df, ref_df


def compute_fingerprints(query_df, ref_df, radius):
    """
    Compute Morgan fingerprints for query and reference molecules.
    
    Returns:
        Tuple of (query_df with fp column, ref_df with fp column)
    """
    logger.info(f"Computing Morgan fingerprints (radius={radius})...")

    try:
        query_df["fp"] = query_df["mol"].apply(
            lambda m: get_morgan_fp(m, radius)
        )
        ref_df["fp"] = ref_df["mol"].apply(
            lambda m: get_morgan_fp(m, radius)
        )
        logger.debug("Fingerprints computed successfully")
    except Exception as e:
        logger.error(f"Error computing fingerprints: {e}")
        raise
    
    return query_df, ref_df


def search_similarities(query_df, ref_df):
    """
    Search all query molecules against reference library.
    
    Returns:
        DataFrame with all query-reference comparisons and similarity scores
    """
    logger.info("Running similarity search...")

    try:
        all_results = []
        ref_fps = ref_df["fp"].tolist()
        
        for query_name, query_fp in zip(query_df["query_name"], query_df["fp"]):
            logger.debug(f"Searching for query molecule: {query_name}")
            
            # Compute similarity scores for this query
            similarities = similarity_search(query_fp, ref_fps)
            
            # Create results dataframe for this query
            query_results = ref_df.copy()
            query_results["query_name"] = query_name
            query_results["similarity"] = similarities
            all_results.append(query_results)
        
        # Combine results from all queries
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.debug(f"Total comparisons: {len(combined_results)} ({len(query_df)} queries × {len(ref_df)} references)")
        return combined_results
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise


def rank_results(combined_results, query_df, top_n):
    """
    Rank results by similarity.
    
    Returns:
        DataFrame with top N hits per query
    """
    logger.info(f"Selecting top {top_n} similar molecules...")

    try:
        # Sort by similarity across all query-reference pairs
        ranked_results = combined_results.sort_values(
            ["query_name", "similarity"],
            ascending=[True, False]
        )
        
        # Get top N per query molecule
        top_hits = ranked_results.groupby("query_name").head(top_n).reset_index(drop=True)
        
        # Keep only relevant columns
        columns_to_keep = ["query_name", "smiles", "ref_name", "similarity"]
        top_hits = top_hits[columns_to_keep]

        logger.info(f"Processed {len(query_df)} query molecules")
        logger.info(f"Found top {top_n} hits per query (total: {len(top_hits)} results)")
        
        # Log stats per query
        for query_name in query_df["query_name"]:
            query_matches = top_hits[top_hits["query_name"] == query_name]
            if len(query_matches) > 0:
                top_sim = query_matches["similarity"].iloc[0]
                bottom_sim = query_matches["similarity"].iloc[-1]
                logger.info(f"  {query_name}: top {top_sim:.3f} → bottom {bottom_sim:.3f}")
        
        return top_hits
    except Exception as e:
        logger.error(f"Error ranking results: {e}")
        raise
