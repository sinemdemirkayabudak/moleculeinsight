"""
Virtual screening pipeline for batch QSAR predictions.

Orchestrates feature computation, QSAR prediction, drug-likeness filtering,
and results ranking for virtual screening workflows.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from app.config import logger
from app.molecule import get_rdkit_properties, lipinski_rules
from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.predict import QSARPredictor
from app.utils import safe_execute


def count_lipinski_violations(mol: Chem.Mol | None) -> int:
    """
    Count number of Lipinski rule-of-5 violations for a molecule.

    Parameters
    ----------
    mol : Chem.Mol | None
        RDKit molecule object or None

    Returns
    -------
    int
        Number of violations (0-4)
    """
    try:
        if mol is None:
            return 4  # Treat None as complete failure
        properties = get_rdkit_properties(mol)
        if not properties:
            return 4  # Treat as complete failure

        rules = safe_execute(lipinski_rules, properties)
        if not rules:
            return 4

        violations = sum(not passed for passed in rules.values())
        return violations
    except Exception as e:
        logger.error(f"Error counting Lipinski violations: {e}")
        return 4  # Treat as complete failure


def compute_qed_score(mol: Chem.Mol | None) -> float | None:
    """
    Compute QED (Quantitative Estimate of Drug-likeness) score.

    QED ranges from 0 to 1, where higher values indicate better drug-likeness.

    Parameters
    ----------
    mol : Chem.Mol | None
        RDKit molecule object or None

    Returns
    -------
    float or None
        QED score (0-1) or None if computation fails
    """
    try:
        return QED.qed(mol)
    except Exception as e:
        logger.warning(f"QED computation failed: {e}")
        return None


def extract_descriptor_values(
    descriptors_matrix: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Extract specific descriptor values from descriptor matrix.

    Parameters
    ----------
    descriptors_matrix : np.ndarray
        RDKit descriptors matrix (n_samples, 8)
    feature_names : list[str]
        Feature names from compute_rdkit_descriptors

    Returns
    -------
    dict
        Dictionary with 'MW' and 'LogP' arrays
    """
    descriptor_dict = {}
    for i, name in enumerate(feature_names):
        descriptor_dict[name] = descriptors_matrix[:, i]

    return {
        "MW": descriptor_dict.get("MW", np.array([])),
        "LogP": descriptor_dict.get("LogP", np.array([])),
    }


def run_virtual_screening_pipeline(
    csv_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Execute complete virtual screening pipeline on batch of molecules.

    Processes: Feature computation → QSAR prediction → Drug-likeness filtering
    → Results ranking by predicted activity (descending).

    Parameters
    ----------
    csv_df : pd.DataFrame
        Input CSV with columns: 'molecule_id', 'smiles'
        Expected dtypes: molecule_id (str), smiles (str)

    Returns
    -------
    dict
        Dictionary with:
        - 'success' (bool): Pipeline completion status
        - 'results' (pd.DataFrame): Screened molecules with predictions
        - 'filtered_out' (int): Molecules failing Lipinski filter
        - 'invalid_smiles' (int): Invalid SMILES count
        - 'total_uploaded' (int): Total input molecules
        - 'final_screened' (int): Passing molecules
        - 'error' (str): Error message if failed

    """
    try:
        logger.info(f"Starting virtual screening pipeline: {len(csv_df)} molecules")

        total_uploaded = len(csv_df)
        invalid_smiles_count = 0
        lipinski_filtered_count = 0

        # Storage for results
        results_data = []

        # Validate SMILES and compute features
        valid_smiles_list = []
        valid_indices = []
        mol_objects = {}

        for idx, row in csv_df.iterrows():
            mol_id = str(row.get("molecule_id", ""))
            smiles = str(row.get("smiles", "")).strip()

            if not smiles:
                invalid_smiles_count += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles_count += 1
                logger.warning(f"Invalid SMILES for {mol_id}: {smiles}")
                continue

            valid_smiles_list.append(smiles)
            valid_indices.append(idx)
            mol_objects[len(valid_smiles_list) - 1] = (mol_id, mol, smiles)

        if not valid_smiles_list:
            logger.error("No valid SMILES found in input")
            return {
                "success": False,
                "results": pd.DataFrame(),
                "invalid_smiles": invalid_smiles_count,
                "total_uploaded": total_uploaded,
                "filtered_out": 0,
                "final_screened": 0,
                "error": "No valid SMILES strings in input CSV",
            }

        logger.info(f"Valid SMILES: {len(valid_smiles_list)}/{total_uploaded}")

        # Compute features
        morgan_result = compute_morgan_fingerprints(valid_smiles_list)
        rdkit_result = compute_rdkit_descriptors(valid_smiles_list)

        if not (morgan_result["success"] and rdkit_result["success"]):
            error_msg = f"Morgan: {morgan_result.get('error', 'Unknown')} | RDKit: {rdkit_result.get('error', 'Unknown')}"
            logger.error(f"Feature computation failed: {error_msg}")
            return {
                "success": False,
                "results": pd.DataFrame(),
                "invalid_smiles": invalid_smiles_count,
                "total_uploaded": total_uploaded,
                "filtered_out": 0,
                "final_screened": 0,
                "error": f"Feature computation failed: {error_msg}",
            }

        X_morgan = morgan_result["X"]
        X_rdkit = rdkit_result["X"]
        rdkit_feature_names = rdkit_result["feature_names"]

        # Combine features
        X_combined = np.hstack([X_morgan, X_rdkit])

        logger.info(f"Features computed. Combined shape: {X_combined.shape}")

        # Load QSAR model and make predictions
        model_path = Path(__file__).parent / "qsar/saved_models/egfr_xgb_model.pkl"

        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return {
                "success": False,
                "results": pd.DataFrame(),
                "invalid_smiles": invalid_smiles_count,
                "total_uploaded": total_uploaded,
                "filtered_out": 0,
                "final_screened": 0,
                "error": "QSAR model not found. Please train the model first.",
            }

        model = joblib.load(model_path)
        y_pred = QSARPredictor.predict(model, X_combined)

        logger.info(f"QSAR predictions completed. Shape: {y_pred.shape}")

        # Extract MW and LogP from descriptors
        descriptor_dict = extract_descriptor_values(X_rdkit, rdkit_feature_names)
        mw_values = descriptor_dict["MW"]
        logp_values = descriptor_dict["LogP"]

        # Process each molecule: QED, Lipinski, extract properties
        for i, (mol_id, mol, smiles) in mol_objects.items():
            # Compute QED score
            qed_score = compute_qed_score(mol)

            # Count Lipinski violations
            lipinski_viols = count_lipinski_violations(mol)

            # Get MW and LogP
            mw = float(mw_values[i]) if i < len(mw_values) else None
            logp = float(logp_values[i]) if i < len(logp_values) else None

            # Get predicted activity
            predicted_activity = float(y_pred[i])

            # Filter by Lipinski: keep if <= 1 violation
            if lipinski_viols > 1:
                lipinski_filtered_count += 1
                continue

            # Add to results
            results_data.append(
                {
                    "Molecule ID": mol_id,
                    "SMILES": smiles,
                    "Predicted Activity (pIC50)": predicted_activity,
                    "QED Score": qed_score if qed_score is not None else np.nan,
                    "Lipinski Violations": lipinski_viols,
                    "MW": mw,
                    "LogP": logp,
                }
            )

        # Create results DataFrame
        results_df = pd.DataFrame(results_data)

        # Sort by predicted activity (descending)
        if len(results_df) > 0:
            results_df = results_df.sort_values(
                "Predicted Activity (pIC50)", ascending=False, ignore_index=True
            )

        final_screened = len(results_df)

        logger.info(
            f"Virtual screening complete: {final_screened} molecules passed filters "
            f"(Lipinski filtered: {lipinski_filtered_count})"
        )

        return {
            "success": True,
            "results": results_df,
            "invalid_smiles": invalid_smiles_count,
            "lipinski_filtered": lipinski_filtered_count,
            "total_uploaded": total_uploaded,
            "final_screened": final_screened,
        }

    except Exception as e:
        logger.exception(f"Virtual screening pipeline error: {e}")
        return {
            "success": False,
            "results": pd.DataFrame(),
            "invalid_smiles": 0,
            "total_uploaded": total_uploaded if "total_uploaded" in locals() else 0,
            "lipinski_filtered": 0,
            "final_screened": 0,
            "error": str(e),
        }
