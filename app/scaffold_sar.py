"""
Scaffold extraction and Structure-Activity Relationship (SAR) analysis.

Extracts Murcko scaffolds from molecules, groups compounds by scaffold,
calculates molecular similarity, and detects activity cliffs (pairs of
similar molecules with large potency differences).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity

from app.config import logger
from app.similarity_search import get_morgan_fp

# Suppress RDKit warnings
warnings.filterwarnings("ignore", category=UserWarning)


def fetch_missing_ic50_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch missing IC50 values from ChEMBL for EGFR using sample data.

    For molecules without standard_value, attempts to match them to the sample
    dataset by structure similarity or returns a status message.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'smiles' column, may have missing 'standard_value'.

    Returns
    -------
    pd.DataFrame
        Dataframe with standard_value column filled where possible.
    """
    if "standard_value" in df.columns and df["standard_value"].notna().all():
        return df

    df_copy = df.copy()

    # Try to load sample data for matching
    sample_file = Path(__file__).parent / "data" / "sample" / "scaffold_sample_ic50.csv"
    if not sample_file.exists():
        logger.warning("Sample data not available for IC50 lookup")
        return df_copy

    try:
        sample_df = pd.read_csv(sample_file)

        # For missing values, try to match by SMILES
        missing_mask = (
            df_copy["standard_value"].isna() if "standard_value" in df_copy.columns else True
        )
        if missing_mask is True or missing_mask.all():
            # All or most values are missing
            logger.info("Attempting to fetch IC50 values for submitted molecules...")

            matched_count = 0
            for idx, row in df_copy.iterrows():
                user_smiles = row.get("smiles")
                if user_smiles and pd.isna(row.get("standard_value")):
                    # Try exact match first
                    match = sample_df[sample_df["smiles"] == user_smiles]
                    if not match.empty:
                        df_copy.at[idx, "standard_value"] = match.iloc[0]["standard_value"]
                        matched_count += 1

            logger.info(f"Matched {matched_count} molecules to sample dataset IC50 values")

            if matched_count == 0:
                logger.warning("No exact SMILES matches found in sample dataset")

    except Exception as e:
        logger.debug(f"Error fetching IC50 values: {e}")

    return df_copy


def extract_murcko_scaffold(smiles: str) -> str | None:
    """
    Extract Murcko scaffold from SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.

    Returns
    -------
    str or None
        Murcko scaffold SMILES or None if extraction fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold_mol = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold_mol if scaffold_mol else None
    except Exception as e:
        logger.debug(f"Failed to extract scaffold from {smiles}: {e}")
        return None


def add_scaffolds_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scaffold column to dataframe with IC50 data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'smiles' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'scaffold' column added, rows with invalid SMILES removed.
    """
    if "smiles" not in df.columns:
        logger.error("Input dataframe must contain 'smiles' column")
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["scaffold"] = df_copy["smiles"].apply(extract_murcko_scaffold)
    df_copy = df_copy.dropna(subset=["scaffold"])

    logger.info(f"Extracted scaffolds for {len(df_copy)} molecules")
    return df_copy


def summarize_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group molecules by scaffold and calculate statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'scaffold' and 'standard_value' (IC50) columns.

    Returns
    -------
    pd.DataFrame
        Scaffold summary with columns: scaffold, molecule_count, avg_activity.
    """
    if "scaffold" not in df.columns or "standard_value" not in df.columns:
        logger.error("Dataframe must contain 'scaffold' and 'standard_value' columns")
        return pd.DataFrame()

    scaffold_summary = (
        df.groupby("scaffold")
        .agg(
            molecule_count=("scaffold", "count"),
            avg_activity=("standard_value", "mean"),
            min_activity=("standard_value", "min"),
            max_activity=("standard_value", "max"),
        )
        .reset_index()
        .sort_values(["molecule_count", "avg_activity"], ascending=[False, True])
    )

    logger.info(f"Summarized {len(scaffold_summary)} unique scaffolds")
    return scaffold_summary


def compute_fingerprints(smiles_list: list[str], radius: int = 2) -> list:
    """
    Compute Morgan fingerprints for multiple SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.
    radius : int
        Neighborhood radius for fingerprint generation (default 2).

    Returns
    -------
    list
        List of fingerprints (None for invalid SMILES).
    """
    mol_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
    fps = [get_morgan_fp(mol, radius) for mol in mol_list]
    logger.info(f"Computed {sum(1 for fp in fps if fp is not None)}/{len(fps)} fingerprints")
    return fps


def compute_tanimoto_similarity(fp1, fp2) -> float | None:
    """
    Compute Tanimoto similarity between two fingerprints.

    Parameters
    ----------
    fp1 : fingerprint object
        First fingerprint.
    fp2 : fingerprint object
        Second fingerprint.

    Returns
    -------
    float or None
        Similarity score (0-1) or None if computation fails.
    """
    if fp1 is None or fp2 is None:
        return None
    try:
        return float(TanimotoSimilarity(fp1, fp2))
    except Exception as e:
        logger.debug(f"Failed to compute similarity: {e}")
        return None


def compute_similarity_matrix(fps: list) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix for fingerprints.

    Parameters
    ----------
    fps : list
        List of fingerprints (from compute_fingerprints).

    Returns
    -------
    np.ndarray
        NxN symmetric similarity matrix.
    """
    n = len(fps)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = compute_tanimoto_similarity(fps[i], fps[j])
            if sim is not None:
                matrix[i, j] = sim
                matrix[j, i] = sim
            else:
                matrix[i, j] = 0.0
                matrix[j, i] = 0.0

    logger.info(f"Computed {n}x{n} similarity matrix")
    return matrix


def detect_activity_cliffs(
    df: pd.DataFrame,
    similarity_threshold: float = 0.85,
    activity_ratio_threshold: float = 100.0,
) -> pd.DataFrame:
    """
    Detect activity cliffs: similar molecules with large potency differences.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'molecule_id', 'smiles', 'standard_value' (IC50) columns.
    similarity_threshold : float
        Minimum Tanimoto similarity to consider (default 0.85).
    activity_ratio_threshold : float
        Minimum IC50 ratio to flag as cliff (default 100.0).

    Returns
    -------
    pd.DataFrame
        Dataframe with activity cliffs: mol1, mol2, similarity, activity_ratio, ic50_molecule_1, ic50_molecule_2.
    """
    if "smiles" not in df.columns or "standard_value" not in df.columns:
        logger.error("Dataframe must contain 'smiles' and 'standard_value' columns")
        return pd.DataFrame()

    smiles_list = df["smiles"].tolist()
    fps = compute_fingerprints(smiles_list)

    sim_matrix = compute_similarity_matrix(fps)

    cliffs = []
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]

            if sim >= similarity_threshold:
                ic50_molecule_1 = df.iloc[i]["standard_value"]
                ic50_molecule_2 = df.iloc[j]["standard_value"]

                # Avoid division by zero
                if ic50_molecule_1 == 0 or ic50_molecule_2 == 0:
                    continue

                ratio = max(ic50_molecule_1, ic50_molecule_2) / min(
                    ic50_molecule_1, ic50_molecule_2
                )

                if ratio >= activity_ratio_threshold:
                    mol_id_a = df.iloc[i].get("molecule_id", f"MOL_{i}")
                    mol_id_b = df.iloc[j].get("molecule_id", f"MOL_{j}")

                    cliffs.append(
                        {
                            "mol1": mol_id_a,
                            "mol2": mol_id_b,
                            "similarity": round(sim, 3),
                            "activity_ratio": round(ratio, 1),
                            "ic50_molecule_1": round(ic50_molecule_1, 2),
                            "ic50_molecule_2": round(ic50_molecule_2, 2),
                            "smiles_a": df.iloc[i]["smiles"],
                            "smiles_b": df.iloc[j]["smiles"],
                        }
                    )

    cliffs_df = pd.DataFrame(cliffs) if cliffs else pd.DataFrame()
    if not cliffs_df.empty:
        cliffs_df = cliffs_df.sort_values("activity_ratio", ascending=False)
    logger.info(f"Detected {len(cliffs_df)} activity cliffs")
    return cliffs_df


def get_ic50_summary_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    """
    Get IC50 data quality and activity statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'standard_value' column.

    Returns
    -------
    dict
        Statistics: total_molecules, with_ic50, missing_ic50, coverage_percent,
                    activity_range, median_ic50, mean_ic50.
    """
    if "standard_value" not in df.columns:
        logger.error("Dataframe must contain 'standard_value' column")
        return {}

    total = len(df)
    with_ic50 = df["standard_value"].notna().sum()
    missing = total - with_ic50

    if with_ic50 > 0:
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        activity_range = f"{df['standard_value'].min():.1f} - {df['standard_value'].max():.1f} nM"
        median_ic50 = float(df["standard_value"].median())
        mean_ic50 = float(df["standard_value"].mean())
    else:
        activity_range = "N/A"
        median_ic50 = 0.0
        mean_ic50 = 0.0

    coverage_percent = (with_ic50 / total * 100) if total > 0 else 0.0

    return {
        "total_molecules": total,
        "with_ic50": with_ic50,
        "missing_ic50": missing,
        "coverage_percent": round(coverage_percent, 1),
        "activity_range": activity_range,
        "median_ic50": median_ic50,
        "mean_ic50": mean_ic50,
    }


def load_sample_ic50_data() -> pd.DataFrame | None:
    """
    Load sample IC50 data from CSV file.

    Returns
    -------
    pd.DataFrame or None
        Sample IC50 dataset (1000 molecules from ChEMBL) or None if loading fails.
    """
    try:
        sample_file = Path(__file__).parent / "data" / "sample" / "scaffold_sample_ic50.csv"
        if not sample_file.exists():
            logger.error(f"Sample data file not found: {sample_file}")
            return None

        df = pd.read_csv(sample_file)
        logger.info(f"Loaded sample IC50 data: {len(df)} molecules")
        return df
    except Exception as e:
        logger.error(f"Error loading sample IC50 data: {e}")
        return None
