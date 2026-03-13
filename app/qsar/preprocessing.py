"""
Data cleaning and preprocessing for QSAR IC50 datasets.

Handles IC50 conversion to pIC50 and outlier removal.
SMILES validation is deferred to feature engineering (features.py).
"""

from typing import Any

import numpy as np
import pandas as pd

from app.config import logger


def convert_ic50_to_pic50(ic50_nm: float) -> float:
    """
    Convert IC50 from nM to pIC50.

    pIC50 = -log10(IC50 in M)
    IC50 in M = IC50 in nM / 1e9

    Parameters
    ----------
    ic50_nm : float
        IC50 value in nanoMolar (nM)

    Returns
    -------
    float
        pIC50 value (higher = more potent)
    """
    if ic50_nm <= 0:
        return np.nan
    ic50_m = ic50_nm / 1e9
    pic50 = -np.log10(ic50_m)
    return pic50


def clean_and_preprocess_data(
    df: pd.DataFrame,
    min_pic50: float = 3.0,
    max_pic50: float = 12.0,
) -> dict[str, Any]:
    """
    Clean and preprocess raw IC50 dataset for QSAR modeling.

    Steps:
    1. Remove rows with missing SMILES
    2. Remove duplicate SMILES
    3. Convert IC50 (nM → M) and calculate pIC50
    4. Remove outliers based on pIC50 range
    5. Reset index

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns: smiles, standard_value, assay_id, reference
    min_pic50 : float
        Minimum pIC50 threshold (default 3.0).
        pIC50=3.0 corresponds to IC50=1 µM (weak binders, removes noise).
        pIC50=7.0 corresponds to IC50=0.1 µM (drug-like potency).
    max_pic50 : float
        Maximum pIC50 threshold (default 12.0).
        pIC50=12.0 corresponds to IC50=1 pM (physical limit, removes artifacts).
        Wider range (3-12) allows high diversity; use 5-10 for stricter filtering.

    Returns
    -------
    dict
        Dictionary with:
        - 'success' (bool): Processing success status
        - 'data' (pd.DataFrame): Cleaned data with columns: smiles, pIC50
        - 'stats' (dict): Processing statistics
        - 'error' (str): Error message if failed

    """

    try:
        logger.info(f"Starting preprocessing on {len(df)} records")
        stats = {
            "input": len(df),
            "removed_missing_smiles": 0,
            "removed_duplicates": 0,
            "removed_invalid_conversions": 0,
            "removed_outliers": 0,
            "output": 0,
        }

        # Step 1: Remove missing SMILES
        initial_count = len(df)
        df = df.dropna(subset=["smiles"])
        stats["removed_missing_smiles"] = initial_count - len(df)
        logger.info(f"After removing missing SMILES: {len(df)} records")

        # Step 2: Remove duplicate SMILES (keep first occurrence)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["smiles"], keep="first")
        stats["removed_duplicates"] = initial_count - len(df)
        logger.info(f"After removing duplicates: {len(df)} records")

        # Step 3: Convert IC50 to pIC50
        df["pIC50"] = df["standard_value"].apply(convert_ic50_to_pic50)

        # Remove rows where pIC50 conversion failed
        initial_count = len(df)
        df = df.dropna(subset=["pIC50"])
        stats["removed_invalid_conversions"] = initial_count - len(df)
        logger.info(f"After pIC50 conversion: {len(df)} records")

        # Step 4: Remove outliers based on pIC50 range
        # Default 3-12 range: min removes millimolar-scale weak binders (noise),
        # max removes picomolar-scale impossible values (measurement artifacts).
        # Customize for stricter filtering: e.g., min_pic50=5.0, max_pic50=10.0
        initial_count = len(df)
        df = df[(df["pIC50"] >= min_pic50) & (df["pIC50"] <= max_pic50)]
        stats["removed_outliers"] = initial_count - len(df)
        logger.info(f"After removing outliers (pIC50 {min_pic50}-{max_pic50}): {len(df)} records")

        # Step 5: Keep only relevant columns
        df = df[["smiles", "pIC50"]].reset_index(drop=True)
        stats["output"] = len(df)

        logger.info(
            f"Preprocessing complete. Output: {len(df)} molecules. "
            f"Removed: {stats['input'] - stats['output']} records"
        )

        return {
            "success": True,
            "data": df,
            "stats": stats,
        }

    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")
        return {
            "success": False,
            "error": str(e),
            "stats": stats,
        }


def get_cleaned_dataset(
    raw_df: pd.DataFrame,
    min_pic50: float = 3.0,
    max_pic50: float = 12.0,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Convenience wrapper for cleaning data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw IC50 data
    min_pic50 : float
        Minimum pIC50 threshold (default 3.0). Use 5.0+ for stricter filtering.
    max_pic50 : float
        Maximum pIC50 threshold (default 12.0). Use 10.0 or lower for stricter filtering.

    Returns
    -------
    tuple
        (cleaned_df, stats) or (None, {}) if failed
    """
    result = clean_and_preprocess_data(raw_df, min_pic50=min_pic50, max_pic50=max_pic50)
    if result["success"]:
        return result["data"], result["stats"]
    return None, result.get("stats", {})
