# ============================================================
# FINGERPRINT GENERATION
# ============================================================

import logging
import warnings

from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

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


def get_morgan_fp(mol, radius):
    """
    Compute Morgan fingerprint using the modern RDKit API.
    Instead of storing the entire chemical structure, a Morgan 
    fingerprint converts a molecule into a binary vector 
    (2048 bits) where each bit represents whether a 
    particular structural feature exists in the molecule.

    Parameters:
        mol (RDKit Mol): The molecule to fingerprint
        radius (int): Neighborhood radius for atom environment detection
            - 0: Only atomic properties (element, charge, etc.)
            - 1: Immediate neighbors (1 bond away)
            - 2: Up to 2 bonds away (ECFP4 standard, recommended)
            - 3+: Extended neighborhoods (more specific/sensitive)

    Returns:
        ExplicitBitVect: A 2048-bit binary fingerprint vector compatible with Tanimoto similarity
    """

    gen = AllChem.GetMorganGenerator(radius=radius) # radius = 2 by default
    # GetFingerprint returns a fingerprint object compatible with TanimotoSimilarity
    return gen.GetFingerprint(mol)


def similarity_search(query_fp, ref_fps):
    """
    Compute Tanimoto similarity between
    query fingerprint and reference fingerprints.
    Tanimoto Similarity = (Number of bits both molecules 
    have as 1 / Number of bits at least one molecule has as 1)

    Returns:
        List of similarity scores. Produce scores 
        from 0.0 (completely different) to 1.0 (identical)
    """
    try:
        return [
            TanimotoSimilarity(query_fp, fp)
            for fp in ref_fps
        ]
    except Exception as e:
        logger.error(f"Error computing similarity scores: {e}")
        raise
