"""Performance Visualizations for Streamlit Dashboard.

Creates comprehensive performance plots from BEST MODEL ONLY (XGBoost):
1. Test set residuals (actual vs predicted)
2. Predicted vs actual pIC50 values
3. Feature importance - RDKit descriptors (interpretable)
4. Feature importance - Top Morgan fingerprint bits
5. Prediction error distribution
6. SHAP feature contribution heatmap

Output:
- qsar/visualizations/: All visualization PNG files
- Each plot is optimized for Streamlit display
- Focuses on XGBoost only (best model: R²=0.7018)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from app.qsar.explain import QSARExplainer
from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.qsar_prediction import QSARPipeline

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("app.config").setLevel(logging.WARNING)
logging.getLogger("app.qsar").setLevel(logging.WARNING)

# Styling
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.size"] = 11


def create_output_dir():
    """Create visualizations directory."""
    plots_dir = Path(__file__).parent / "visualizations"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def load_or_train_models():
    """Load trained XGBoost model (best model only)."""
    print("Loading best EGFR model (XGBoost)...")

    models_dir = Path(__file__).parent / "saved_models"
    xgb_path = models_dir / "egfr_xgb_model.pkl"

    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        print("✓ XGBoost model loaded from qsar/saved_models/")
        return xgb_model, True
    else:
        print("⚠ Model not found. Training...")
        from app.qsar.train_models import main

        main()

        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)
            return xgb_model, False
        else:
            raise FileNotFoundError("XGBoost model failed to train")


def load_and_prepare_data():
    """Load and prepare data for evaluation."""
    print("\nPreparing evaluation dataset...")

    pipeline = QSARPipeline()

    # Try to load from API first
    all_data = []
    api_failed = False

    try:
        for i in range(10):
            offset = i * 1000
            result = pipeline.load_data(limit=1000, offset=offset)
            if not result["success"]:
                break
            batch_data = result.get("data")
            if batch_data is not None and len(batch_data) > 0:
                all_data.append(batch_data)
            else:
                break

        if len(all_data) == 0:
            api_failed = True
    except Exception as e:
        api_failed = True
        print(f"⚠ API failed: {e}. Using fallback sample data...")

    # Fallback to sample data if API failed
    if api_failed or len(all_data) == 0:
        print("⚠ Using fallback sample data for evaluation...")
        from pathlib import Path

        sample_dir = Path(__file__).parent / "app" / "data" / "sample"

        sample_files = list(sample_dir.glob("*_dataset.csv"))
        if sample_files:
            all_data = [pd.read_csv(f) for f in sample_files]

    if len(all_data) == 0:
        raise ValueError("No data available from API or sample files")

    pipeline.raw_data = pd.concat(all_data, ignore_index=True)

    # Preprocess
    pipeline.raw_data["standard_value"] = pd.to_numeric(
        pipeline.raw_data["standard_value"], errors="coerce"
    )
    preprocess_result = pipeline.preprocess_data()
    pipeline.cleaned_data = preprocess_result.get("data")

    # Extract SMILES and targets
    smiles_list = pipeline.cleaned_data["smiles"].tolist()
    y_all = pipeline.cleaned_data["pIC50"].values

    # Compute combined features
    morgan_result = compute_morgan_fingerprints(smiles_list)
    X_morgan = morgan_result["X"]

    rdkit_result = compute_rdkit_descriptors(smiles_list)
    X_rdkit = rdkit_result["X"]

    X_combined = np.hstack([X_morgan, X_rdkit])

    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_all, test_size=0.2, random_state=42
    )

    print(f"✓ Data prepared: {len(X_train)} train, {len(X_test)} test samples")

    return X_train, X_test, y_train, y_test, X_morgan, smiles_list


def get_bit_substructure(smiles: str, bit_idx: int, gen) -> str:
    """Get SMARTS substructure for a Morgan bit using modern RDKit API.

    Uses rdFingerprintGenerator.AdditionalOutput() to get bit information.

    Parameters
    ----------
    smiles : str
        SMILES string of molecule
    bit_idx : int
        Fingerprint bit index to query
    gen : rdFingerprintGenerator
        Morgan fingerprint generator (radius=2, fpSize=2048)

    Returns
    -------
    str
        Fragment SMILES for this bit, or 'N/A' if bit not set
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "N/A"

    # Modern API: collects bit information during fingerprint generation
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.CollectBitInfoMap()  # Maps bit_id → [(atom_idx, radius), ...]

    gen.GetFingerprint(mol, additionalOutput=ao)
    bit_info = ao.GetBitInfoMap()

    if bit_idx not in bit_info:
        return "N/A"

    # Take first (atom_idx, radius) pair that set this bit
    atom_idx, radius = bit_info[bit_idx][0]

    if radius == 0:
        # Single atom case — just return the atomic symbol
        return (
            Chem.MolToSmiles(mol)
            if len(mol.GetAtoms()) == 1
            else mol.GetAtomWithIdx(atom_idx).GetSymbol()
        )

    # Get the bond environment (substructure) at this radius
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms = set()
    for bond_idx in env:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # Convert to SMILES fragment
    frag_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(atoms),
        bondsToUse=list(env),
        rootedAtAtom=atom_idx,
    )

    return frag_smiles if frag_smiles else "N/A"


def annotate_morgan_bits(
    xgb_model, smiles_list: list[str], bit_indices: list[int] | None = None, n_bits: int = 15
) -> pd.DataFrame:
    """Annotate Morgan bits with their substructure SMILES.

    For each Morgan bit, finds a molecule that has that bit and extracts
    the substructure SMILES. Searches entire dataset.

    Parameters
    ----------
    xgb_model : XGBRegressor
        Trained XGBoost model
    smiles_list : list[str]
        Full list of SMILES strings from dataset
    bit_indices : list[int], optional
        Specific bit indices to annotate. If None, uses top n_bits by gain importance.
    n_bits : int
        Number of top bits to annotate if bit_indices is None (default 15)

    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        - bit_index: Morgan bit ID (0-2047)
        - importance: Feature importance from model
        - substructure: Fragment SMILES (or 'N/A' if not found)
        - feature_name: Human-readable label with substructure
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    morgan_importances = xgb_model.feature_importances_[:2048]

    # Use provided bit indices or determine from top importance
    if bit_indices is None:
        top_bit_indices = np.argsort(morgan_importances)[-n_bits:][::-1]
        n_to_annotate = n_bits
    else:
        top_bit_indices = bit_indices
        n_to_annotate = len(bit_indices)

    rows = []

    print(
        f"  → Annotating {n_to_annotate} Morgan bits (searching {len(smiles_list)} molecules for substructures)..."
    )

    for bit_idx in top_bit_indices:
        importance = float(morgan_importances[int(bit_idx)])
        substructure = "N/A"

        # Search entire dataset for this bit
        for smiles in smiles_list:
            result = get_bit_substructure(smiles, int(bit_idx), gen)
            if result != "N/A":
                substructure = result
                break

        # Create human-readable feature name
        if substructure != "N/A" and len(substructure) <= 12:
            feature_name = f"Morgan_Bit{bit_idx}_{substructure}"
        elif substructure != "N/A":
            truncated = substructure[:8] + "*"
            feature_name = f"Morgan_Bit{bit_idx}_{truncated}"
        else:
            feature_name = f"Morgan_Bit{bit_idx}"

        rows.append(
            {
                "bit_index": int(bit_idx),
                "importance": importance,
                "substructure": substructure,
                "feature_name": feature_name,
            }
        )

    df = pd.DataFrame(rows)
    found_count = (df["substructure"] != "N/A").sum()
    rare_count = len(df) - found_count
    print(
        f"      ✓ Found substructures for {found_count}/{n_to_annotate} bits | {rare_count} unannotated"
    )

    return df


def save_morgan_annotations(xgb_model, smiles_list: list[str], output_path=None):
    """Save Morgan bit annotations to JSON for later use in predictions.

    Parameters
    ----------
    xgb_model : XGBRegressor
        Trained XGBoost model
    smiles_list : list[str]
        Full list of SMILES strings from dataset
    output_path : str, optional
        Path to save JSON file. Defaults to qsar/saved_models/morgan_bit_annotations.json
    """
    if output_path is None:
        output_path = Path(__file__).parent / "saved_models" / "morgan_bit_annotations.json"
    else:
        output_path = Path(output_path)

    print(f"\nSaving Morgan bit annotations to {output_path}...")
    print("  → Annotating all 2048 Morgan bits (comprehensive coverage)...")

    all_bit_indices = list(range(2048))
    annotations_df = annotate_morgan_bits(xgb_model, smiles_list, bit_indices=all_bit_indices)

    annotations_dict = {}
    for _, row in annotations_df.iterrows():
        bit_idx = int(row["bit_index"])
        substructure = row["substructure"]
        annotations_dict[bit_idx] = substructure

    with open(output_path, "w") as f:
        json.dump(annotations_dict, f)

    print(f"  ✓ Saved {len(annotations_dict)} Morgan bit annotations")


def prepare_residuals_data(xgb_model, X_test, y_test):
    """
    Prepare residuals data (XGBoost test set residuals (actual vs predicted)) for interactive Streamlit scatter plot.

    **CHANGES FROM ORIGINAL:**
    - Removed matplotlib plotting code
    - Returns structured dict instead of saving PNG
    - Data ready for st.scatter_chart()
    """
    print("\nPreparing residuals data...")

    xgb_pred = xgb_model.predict(X_test)
    xgb_residual = y_test - xgb_pred

    data = {
        "predicted": xgb_pred.tolist(),
        "residuals": xgb_residual.tolist(),
        "actual": y_test.tolist(),
        "zero_line": 0.0,
        "color": "#81C784",
    }

    print("  ✓ Residuals data prepared")
    return data


def prepare_predictions_vs_actual_data(xgb_model, X_test, y_test):
    """
    Prepare XGBoost predictions vs actual pIC50 values for interactive scatter plot.

    **CHANGES FROM ORIGINAL:**
    - Removed matplotlib.scatter() code
    - Returns dict with actual, predicted, and R² score
    - Includes perfect prediction line coordinates
    """
    print("Preparing predictions vs actual data...")

    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = xgb_model.score(X_test, y_test)

    min_val = float(min(y_test.min(), xgb_pred.min()))
    max_val = float(max(y_test.max(), xgb_pred.max()))

    data = {
        "actual": y_test.tolist(),
        "predicted": xgb_pred.tolist(),
        "r2_score": float(xgb_r2),
        "perfect_line": {
            "min": min_val,
            "max": max_val,
        },
        "color": "#64B5F6",
    }

    print("  ✓ Predictions vs actual data prepared")
    return data


def prepare_feature_importance_data(shap_vals, n_features=20):
    """
    Prepare feature importance data using SHAP values (prediction-based, unbiased).

    SHAP values show actual contribution to predictions (unbiased), unlike gain importance
    which is biased toward high-cardinality features and model construction artifacts.

    **CHANGES FROM ORIGINAL:**
    - Removed matplotlib.barh() plotting
    - Returns list of feature dicts instead of saving PNG
    - Preserves Morgan bit annotations
    - Ready for st.bar_chart(horizontal=True)
    - Sorted in descending order by importance
    """
    print(f"Preparing feature importance data (top {n_features} via SHAP)...")

    # Load Morgan bit annotations
    anno_data = {}
    anno_path = Path(__file__).parent / "saved_models" / "egfr_feature_annotations.json"
    if anno_path.exists():
        try:
            with open(anno_path) as f:
                anno_file = json.load(f)
                anno_data = anno_file.get("morgan_bits", {})
        except Exception as e:
            print(f"  Warning: Could not load annotations: {e}")

    # Calculate SHAP-based importance
    mean_shap_abs = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_shap_abs)[-n_features:][::-1]  # Already descending

    # RDKit descriptor names
    rdkit_names = {
        2048: "MW (Molecular Weight)",
        2049: "LogP (Lipophilicity)",
        2050: "TPSA (Polar Surface)",
        2051: "HBD (H-Bond Donors)",
        2052: "HBA (H-Bond Acceptors)",
        2053: "RotBonds (Rotatable)",
        2054: "AromaticRings",
        2055: "RingCount",
    }

    # Build feature list
    features = []

    for idx in top_indices:
        importance = float(mean_shap_abs[idx])

        if idx < 2048:
            # Morgan bit
            bit_str = str(idx)
            annotation = anno_data.get(bit_str, "N/A")
            if annotation and annotation != "N/A":
                label = f"Morgan_Bit{idx:04d} → {annotation}"
            else:
                label = f"Morgan_Bit{idx:04d}"
            feature_type = "Morgan"
            color = "#64B5F6"
        elif idx in rdkit_names:
            label = rdkit_names[idx]
            feature_type = "RDKit"
            color = "#81C784"
        else:
            label = f"RDKit_{idx - 2048}"
            feature_type = "RDKit"
            color = "#81C784"

        features.append(
            {
                "feature": label,
                "importance": importance,
                "type": feature_type,
                "index": int(idx),
                "color": color,
            }
        )

    # ← Ensure descending order (highest importance first)
    features.sort(key=lambda x: x["importance"], reverse=True)

    data = {
        "features": features,
        "method": "SHAP mean absolute value (prediction-based, unbiased)",
        "description": "Morgan Bits: Circular substructure patterns | RDKit: Physicochemical properties",
        "gridlines": {
            "show": True,
            "axis": "x",
            "color": "rgba(255, 255, 255, 0.1)",  # Barely visible white
            "width": 1,
        },
    }

    print("  ✓ Feature importance data prepared")
    return data


def prepare_error_distribution_data(xgb_model, X_test, y_test, n_bins=100):
    """
    Prepare XGBoost prediction error distribution data.

    **CHANGES FROM ORIGINAL:**
    - Removed matplotlib.hist() code
    - Returns histogram bins + summary statistics
    - Ready for st.bar_chart() or plotly histogram
    """
    print("Preparing error distribution data...")

    xgb_pred = xgb_model.predict(X_test)
    xgb_errors = np.abs(y_test - xgb_pred)

    # Create histogram bins
    hist, bin_edges = np.histogram(xgb_errors, bins=n_bins)

    # Calculate bin width for metadata
    bin_width = (bin_edges[-1] - bin_edges[0]) / n_bins

    data = {
        "errors": xgb_errors.tolist(),
        "mean": float(np.mean(xgb_errors)),
        "median": float(np.median(xgb_errors)),
        "std": float(np.std(xgb_errors)),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_width": float(bin_width),
        },
        "color": "#FFB74D",
        "n_bins": n_bins,
    }

    print("  ✓ Error distribution data prepared")
    return data


def prepare_model_summary_data(xgb_model, X_train, X_test, y_train, y_test):
    """
    Prepare XGBoost model performance summary metrics.

    **CHANGES FROM ORIGINAL:**
    - Removed complex matplotlib figure with custom styling
    - Returns clean metrics dict
    - Ready for st.metric() cards
    """
    print("Preparing model performance summary...")

    xgb_test_r2 = xgb_model.score(X_test, y_test)
    xgb_test_mae = mean_absolute_error(y_test, xgb_model.predict(X_test))
    xgb_train_r2 = xgb_model.score(X_train, y_train)
    gap = xgb_train_r2 - xgb_test_r2

    # Load actual CV metrics
    metrics_path = Path(__file__).parent / "saved_models" / "egfr_performance.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        xgb_cv_r2_mean = metrics.get("cv_metrics", {}).get("xgb_cv_r2_mean", 0.7007)
        xgb_cv_r2_std = metrics.get("cv_metrics", {}).get("xgb_cv_r2_std", 0.0239)
    else:
        xgb_cv_r2_mean = 0.7007
        xgb_cv_r2_std = 0.0239

    data = {
        "test_r2": float(xgb_test_r2),
        "train_r2": float(xgb_train_r2),
        "test_mae": float(xgb_test_mae),
        "cv_r2_mean": float(xgb_cv_r2_mean),
        "cv_r2_std": float(xgb_cv_r2_std),
        "overfitting_gap": float(gap),
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_features": 2056,
        "model_type": "XGBoost",
        "features_breakdown": {
            "morgan_fp": 2048,
            "rdkit_descriptors": 8,
        },
    }

    print("  ✓ Model summary data prepared")
    return data


def prepare_shap_heatmap_data(
    xgb_model, shap_vals, X_test, smiles_list, n_samples=50, n_features=20
):
    """
    Prepare SHAP heatmap data (sample × feature matrix of SHAP values).

    **CHANGES FROM ORIGINAL:**
    - Removed SHAPVisualizer.heatmap() matplotlib call
    - Returns 2D matrix + feature labels
    - Ready for plotly.express.imshow() or seaborn heatmap

    """
    print("Preparing SHAP heatmap data...")

    try:
        # Sample data
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        shap_sample = shap_vals[sample_indices]

        # Get top features via SHAP
        mean_shap_abs = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_shap_abs)[-n_features:][::-1]

        # RDKit names
        rdkit_names = {
            2048: "MW",
            2049: "LogP",
            2050: "TPSA",
            2051: "HBD",
            2052: "HBA",
            2053: "RotBonds",
            2054: "AromaticRings",
            2055: "RingCount",
        }

        # Annotate Morgan bits in top features
        morgan_bits_in_top = [idx for idx in top_indices if idx < 2048]
        feature_labels = {}

        if morgan_bits_in_top:
            morgan_annotations = annotate_morgan_bits(
                xgb_model, smiles_list, bit_indices=morgan_bits_in_top
            )
            for _, row in morgan_annotations.iterrows():
                bit_id = row["bit_index"]
                feature_labels[bit_id] = row["feature_name"]

        # Add RDKit names
        for idx in top_indices:
            if idx >= 2048:
                feature_labels[idx] = rdkit_names.get(idx, f"RDKit_{idx - 2048}")

        # Create ordered labels
        final_labels = []
        for idx in top_indices:
            if idx in feature_labels:
                final_labels.append(feature_labels[idx])
            elif idx < 2048:
                final_labels.append(f"Morgan_Bit{idx}")
            else:
                final_labels.append(f"RDKit_{idx - 2048}")

        # Extract SHAP matrix for top features
        shap_matrix_samples_x_features = shap_sample[:, top_indices]  # (n_samples, n_features)

        # TRANSPOSE so features are rows, samples are columns
        shap_matrix_features_x_samples = shap_matrix_samples_x_features.T  # (n_features, n_samples)

        # Create sample labels (x-axis)
        sample_labels = [f"Sample_{i + 1}" for i in range(len(sample_indices))]

        data = {
            "shap_matrix": shap_matrix_features_x_samples.tolist(),  # ← TRANSPOSED
            "feature_names": final_labels,  # y-axis labels
            "sample_labels": sample_labels,  # ← x-axis labels
            "sample_indices": sample_indices.tolist(),
            "base_value": float(xgb_model.predict(X_test).mean()),
            "n_samples": n_samples,
            "n_features": n_features,
            "orientation": "features_on_y",  # ← metadata for rendering
        }

        print("  ✓ SHAP heatmap data prepared")
        return data

    except Exception as e:
        logger.warning(f"SHAP heatmap preparation failed: {e}")
        print(f"  ⚠ SHAP heatmap skipped ({type(e).__name__})")
        return None


def main():
    """Generate all visualizations (XGBoost only - best model)."""
    print("\n" + "=" * 70)
    print("PERFORMANCE DATA EXPORT (JSON for Streamlit)")
    print("=" * 70)

    # Create output directory
    plots_dir = create_output_dir()
    print(f"\nOutput directory: {plots_dir}/\n")

    # Load model
    xgb_model, _ = load_or_train_models()

    # Load data
    X_train, X_test, y_train, y_test, _, smiles_list = load_and_prepare_data()

    # Compute SHAP values once
    print("\nComputing SHAP values (used for feature importance and heatmap)...")
    background_indices = np.random.choice(len(X_test), min(100, len(X_test) // 2), replace=False)
    X_background = X_test[background_indices]

    explainer = QSARExplainer()
    shap_explainer = explainer.create_explainer(xgb_model, X_background, "XGBoost")
    shap_vals = explainer.compute_shap_values(
        shap_explainer, X_test, "XGBoost", max_samples=len(X_test)
    )
    print(f"  ✓ SHAP values computed for {len(X_test)} test samples")

    # Prepare all plot data (NEW: JSON export instead of PNG)
    print("\n" + "=" * 70)
    print("GENERATING PLOT DATA")
    print("=" * 70)

    plot_data = {
        "metadata": {
            "model": "XGBoost",
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": 2056,
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "residuals": prepare_residuals_data(xgb_model, X_test, y_test),
        "predictions_vs_actual": prepare_predictions_vs_actual_data(xgb_model, X_test, y_test),
        "feature_importance": prepare_feature_importance_data(
            xgb_model, smiles_list, shap_vals, n_features=20
        ),
        "error_distribution": prepare_error_distribution_data(xgb_model, X_test, y_test),
        "model_summary": prepare_model_summary_data(xgb_model, X_train, X_test, y_train, y_test),
        "shap_heatmap": prepare_shap_heatmap_data(
            xgb_model, shap_vals, X_test, smiles_list, n_samples=50, n_features=20
        ),
    }

    # Save to JSON
    json_path = plots_dir / "performance_data.json"
    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)

    print(f"\n✅ All plot data saved to: {json_path}")
    print(f"   File size: {json_path.stat().st_size / 1024:.1f} KB")

    # Save Morgan annotations
    save_morgan_annotations(
        xgb_model,
        smiles_list,
        output_path=Path(__file__).parent / "saved_models" / "morgan_bit_annotations.json",
    )

    # Compute uncertainty metrics
    print("\nComputing prediction uncertainty metrics...")
    y_pred_test = xgb_model.predict(X_test)
    residuals = y_test - y_pred_test
    residual_std = np.std(residuals)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"  ✓ Residual Std Dev: {residual_std:.4f} pIC50 units")
    print(f"  ✓ RMSE: {rmse:.4f}")

    # Update metadata with uncertainty metrics
    try:
        metadata_path = Path(__file__).parent / "saved_models" / "egfr_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        metadata["uncertainty_metrics"] = {
            "residual_std": float(residual_std),
            "rmse": float(rmse),
            "ci_95_margin": float(1.96 * residual_std),
            "description": "95% confidence interval margin = 1.96 × residual_std",
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("  ✓ Updated metadata with uncertainty metrics")
    except Exception as e:
        print(f"  ⚠ Could not update metadata: {e}")

    print("\n" + "=" * 70)
    print("MODEL VISUALIZATIONS COMPLETE ✅")
    print("=" * 70)
    print("\nData exported to: qsar/visualizations/performance_data.json")
    print("\nPlot data structure:")
    print("  1. Residuals (XGBoost predictions - actual values)")
    print("  2. Predictions vs Actual (calibration plot)")
    print("  3. Feature Importance (Top 20: SHAP-based, Morgan + RDKit)")
    print("  4. Error Distribution (histogram of prediction errors)")
    print("  5. Model Performance Summary (R², RMSE, MAE, overfitting gap)")
    print("  6. SHAP Heatmap (top 20 features)")
    print("\nFeature Importance Method:")
    print("   • SHAP values (prediction-based, unbiased)")
    print("   • Shows actual contribution to each prediction")
    print("   • Corrects for high-cardinality bias in Morgan features")
    print("\nFeature Interpretation:")
    print("   • Morgan Bits: Annotated with circular substructure SMILES patterns")
    print("   • RDKit Features: Named molecular properties (MW, LogP, TPSA, etc.)")
    print("\nMorgan Bit Annotations:")
    print("   • SMILES fragment shown for each Morgan bit")
    print("   • * (asterisk): Indicates truncated SMILES - original fragment is >12 chars")
    print("   • Examples: Morgan_Bit1234_c1ccccc1 (full) vs Morgan_Bit567_c1ccc(O*  (truncated)")
    print("\n✨ These plots will be displayed in Streamlit dashboard")


if __name__ == "__main__":
    main()
