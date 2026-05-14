"""
Feature engineering for QSAR models.

Converts SMILES strings to Morgan fingerprints or RDKit descriptors.
"""

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from app.config import logger


def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
) -> dict[str, Any]:
    """
    Compute Morgan fingerprints for list of SMILES.

    Morgan fingerprints (circular fingerprints) are widely used in QSAR
    for capturing molecular structural patterns. Uses 2048 bits by default.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings
    radius : int
        Radius for circular fingerprints (default 2)

    Returns
    -------
    dict
        Dictionary with:
        - 'success' (bool): Computation success
        - 'X' (np.ndarray): Feature matrix (n_samples, 2048)
        - 'feature_names' (list): Fingerprint bit indices
        - 'error' (str): Error message if failed

    """

    try:
        logger.info(
            f"Computing Morgan fingerprints for {len(smiles_list)} "
            f"molecules (2048 bits, radius {radius})"
        )

        fingerprints = []
        failed = 0

        # Create Morgan generator once (more efficient)
        gen = AllChem.GetMorganGenerator(radius=radius)  # type: ignore

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed += 1
                    continue

                # Use GetFingerprintAsNumPy for direct numpy array conversion
                fp = gen.GetFingerprintAsNumPy(mol)
                fingerprints.append(fp)
            except Exception:
                failed += 1
                continue

        if not fingerprints:
            logger.error("No valid fingerprints computed")
            return {"success": False, "error": "No valid fingerprints"}

        X = np.array(fingerprints)  # Shape: (n_samples, 2048) with 0s and 1s
        # X: Feature matrix where each row is a molecule, each column is a fingerprint bit
        # feature_names: Labels only (metadata), like column headers
        # X[i, j] = 1 if bit j is present in molecule i, else 0
        # Example: X[0, 5] = 1 means "bit_5" is present in first molecule
        # So if you have 100 molecules: X shape is (100, 2048)
        feature_names = [f"bit_{i}" for i in range(2048)]

        logger.info(
            f"Computed {len(fingerprints)} Morgan fingerprints. Failed: {failed}/{len(smiles_list)}"
        )

        return {
            "success": True,
            "X": X,
            "feature_names": feature_names,
        }

    except Exception as e:
        logger.exception(f"Error computing Morgan fingerprints: {e}")
        return {"success": False, "error": str(e)}


def compute_rdkit_descriptors(
    smiles_list: list[str],
) -> dict[str, Any]:
    """
    Compute RDKit molecular descriptors for list of SMILES.

    Includes: MW, LogP, HBD, HBA, TPSA, RotBonds, AromaticRings, etc.
    Useful for understanding physicochemical properties in QSAR.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings

    Returns
    -------
    dict
        Dictionary with:
        - 'success' (bool): Computation success
        - 'X' (np.ndarray): Feature matrix (n_samples, n_descriptors)
        - 'feature_names' (list): Descriptor names
        - 'error' (str): Error message if failed

    """

    # Define RDKit descriptors to compute
    descriptor_list = [
        ("MW", Descriptors.MolWt),  # type: ignore  # Molecular Weight
        ("LogP", Descriptors.MolLogP),  # type: ignore  # Partition coefficient (lipophilicity)
        ("HBD", Descriptors.NumHDonors),  # type: ignore  # Hydrogen Bond Donors
        ("HBA", Descriptors.NumHAcceptors),  # type: ignore  # Hydrogen Bond Acceptors
        ("TPSA", Descriptors.TPSA),  # type: ignore  # Topological Polar Surface Area
        ("RotBonds", Descriptors.NumRotatableBonds),  # type: ignore  # Rotatable Bonds
        ("AromaticRings", Descriptors.NumAromaticRings),  # type: ignore  # Aromatic Ring Count
        ("RingCount", Descriptors.RingCount),  # type: ignore  # Total Ring Count
    ]

    try:
        logger.info(f"Computing RDKit descriptors for {len(smiles_list)} molecules")

        descriptors_data = []
        failed = 0

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed += 1
                    continue

                desc_values = []
                for _, desc_func in descriptor_list:
                    try:
                        value = desc_func(mol)
                        desc_values.append(value)
                    except Exception:
                        desc_values.append(np.nan)

                descriptors_data.append(desc_values)
            except Exception:
                failed += 1
                continue

        if not descriptors_data:
            logger.error("No valid descriptors computed")
            return {"success": False, "error": "No valid descriptors"}

        X = np.array(descriptors_data)  # Shape: (n_samples, 8) with numerical values
        # X: Feature matrix where each row is a molecule, each column is a descriptor
        # feature_names: Labels only (metadata), like column headers
        # X[i, j] = numerical value of descriptor j for molecule i
        # Example: X[0, 0] = 256.3 means MW (feature_names[0]) is 256.3 for first molecule
        # X[1, 2] = 3 means HBD (feature_names[2]) is 3 for second molecule
        # So if you have 100 molecules: X shape is (100, 8)
        feature_names = [name for name, _ in descriptor_list]

        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in descriptors")

        logger.info(
            f"Computed {len(descriptors_data)} RDKit descriptor sets. "
            f"Features: {len(feature_names)}. Failed: {failed}/{len(smiles_list)}"
        )

        return {
            "success": True,
            "X": X,
            "feature_names": feature_names,
        }

    except Exception as e:
        logger.exception(f"Error computing RDKit descriptors: {e}")
        return {"success": False, "error": str(e)}


def prepare_features(
    df: pd.DataFrame,
    feature_type: str = "morgan",
) -> dict[str, Any]:
    """
    Prepare ML features from cleaned QSAR dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with columns: smiles, pIC50
    feature_type : str
        Feature type: 'morgan' or 'descriptors' (default 'morgan')

    Returns
    -------
    dict
        Dictionary with:
        - 'success' (bool): Feature computation success
        - 'X' (np.ndarray): Feature matrix
        - 'y' (np.ndarray): Target values (pIC50)
        - 'feature_names' (list): Feature names
        - 'smiles' (np.ndarray): Original SMILES
        - 'error' (str): Error message if failed

    """

    try:
        if feature_type.lower() == "morgan":
            result = compute_morgan_fingerprints(df["smiles"].tolist())
        elif feature_type.lower() == "descriptors":
            result = compute_rdkit_descriptors(df["smiles"].tolist())
        else:
            return {
                "success": False,
                "error": f"Unknown feature type: {feature_type}",
            }

        if not result.get("success"):
            return result

        # X: Feature matrix (rows=molecules, columns=features)
        # For 'morgan': (n_samples, 2048) with 0s and 1s (structural patterns)
        # For 'descriptors': (n_samples, 8) with numerical values (physicochemical properties)
        X = result["X"]
        y = df["pIC50"].values[: len(X)]  # Match X length
        smiles = df["smiles"].values[: len(X)]

        logger.info(
            f"Features prepared: X shape {X.shape}, y shape {y.shape}, feature_type={feature_type}"
        )

        return {
            "success": True,
            "X": X,
            "y": y,
            "smiles": smiles,
            "feature_names": result["feature_names"],
            "feature_type": feature_type,
        }

    except Exception as e:
        logger.exception(f"Error preparing features: {e}")
        return {"success": False, "error": str(e)}
