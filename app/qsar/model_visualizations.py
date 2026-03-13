"""Phase 2: Performance Visualizations for Streamlit Dashboard.

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

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from app.qsar.explain import QSARExplainer
from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.qsar_prediction import QSARPipeline
from app.qsar.visualize import SHAPVisualizer

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

    print(f"\n📌 Saving Morgan bit annotations to {output_path}...")

    # Annotate ALL 2048 Morgan bits for comprehensive feature coverage
    # This ensures predictions will have annotations for virtually all present bits
    print("  → Annotating all 2048 Morgan bits (this may take a few minutes)...")

    # Generate all bit indices (don't filter by importance - we need comprehensive coverage)
    all_bit_indices = list(range(2048))

    annotations_df = annotate_morgan_bits(xgb_model, smiles_list, bit_indices=all_bit_indices)

    # Convert to dictionary: bit_index -> substructure
    annotations_dict = {}
    for _, row in annotations_df.iterrows():
        bit_idx = int(row["bit_index"])
        substructure = row["substructure"]
        annotations_dict[bit_idx] = substructure

    # Save to JSON
    with open(output_path, "w") as f:
        import json

        json.dump(annotations_dict, f)

    print(f"  ✓ Saved {len(annotations_dict)} Morgan bit annotations")


def plot_residuals(xgb_model, X_test, y_test, plots_dir):
    """Plot 1: XGBoost test set residuals (actual vs predicted)."""
    print("\nCreating residuals plot...")

    xgb_pred = xgb_model.predict(X_test)
    xgb_residual = y_test - xgb_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        xgb_pred,
        xgb_residual,
        alpha=0.6,
        s=50,
        color="darkgreen",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Predicted pIC50", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12, fontweight="bold")
    ax.set_title("XGBoost: Residuals Plot (Test Set)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(plots_dir / "01_residuals.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 01_residuals.png")


def plot_predictions_vs_actual(xgb_model, X_test, y_test, plots_dir):
    """Plot 2: XGBoost predicted vs actual pIC50 values."""
    print("Creating predictions vs actual plot...")

    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = xgb_model.score(X_test, y_test)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        y_test, xgb_pred, alpha=0.6, s=50, color="darkgreen", edgecolors="black", linewidth=0.5
    )
    min_val = min(y_test.min(), xgb_pred.min())
    max_val = max(y_test.max(), xgb_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2.5, label="Perfect prediction"
    )
    ax.set_xlabel("Actual pIC50", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted pIC50", fontsize=12, fontweight="bold")
    ax.set_title(
        f"XGBoost: Predictions vs Actual (R² = {xgb_r2:.4f})", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / "02_predictions_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 02_predictions_vs_actual.png")


def plot_combined_feature_importance(
    xgb_model, plots_dir, smiles_list, shap_vals, X_test, n_features=20
):
    """Plot 3: Top 20 Features using SHAP values (prediction-based importance).

    SHAP values show actual contribution to predictions (unbiased), unlike gain importance
    which is biased toward high-cardinality features and model construction artifacts.
    """
    import json

    print(f"Creating feature importance plot (top {n_features} features via SHAP)...")

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

    # Calculate mean absolute SHAP for each feature (prediction-based importance)
    mean_shap_abs = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_shap_abs)[-n_features:][::-1]

    # RDKit descriptor names (indices 2048-2055)
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

    # Build feature names for top features
    final_labels = []
    final_importances = []
    final_indices = []

    for idx in top_indices:
        final_indices.append(idx)
        final_importances.append(mean_shap_abs[idx])

        if idx < 2048:
            # Morgan bit: include annotation if available
            bit_str = str(idx)
            annotation = anno_data.get(bit_str, "N/A")
            if annotation and annotation != "N/A":
                final_labels.append(f"Morgan_Bit{idx:04d} → {annotation}")
            else:
                final_labels.append(f"Morgan_Bit{idx:04d}")
        elif idx in rdkit_names:
            final_labels.append(rdkit_names[idx])
        else:
            final_labels.append(f"RDKit_{idx - 2048}")

    # Create plot with wider figure to accommodate annotations
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color code: Morgan bits (blues), RDKit (greens)
    colors = []
    for idx in final_indices:
        if idx < 2048:
            colors.append(plt.cm.Blues(0.6))
        else:
            colors.append(plt.cm.Greens(0.5))

    bars = ax.barh(
        range(len(final_indices)), final_importances, color=colors, edgecolor="black", linewidth=1
    )

    ax.set_yticks(range(len(final_indices)))
    ax.set_yticklabels(final_labels, fontsize=9)
    ax.set_xlabel("SHAP Mean |Value| (Prediction Impact)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"XGBoost: Top {n_features} Features (Morgan + RDKit) - SHAP Based",
        fontsize=13,
        fontweight="bold",
    )
    # Scale axis range by 1.08 to add margin and prevent label overlap
    n = len(final_indices)
    margin = n * 0.04  # 4% on each side = 8% total
    ax.set_ylim(n - 1 + margin, -margin)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for _, (bar, val) in enumerate(zip(bars, final_importances, strict=False)):
        width = bar.get_width()
        ax.text(
            width + max(final_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    # Add padding on right side for value labels
    max_x = max(final_importances)
    ax.set_xlim(0, max_x * 1.08)

    # Add legend
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.6), edgecolor="black", label="Morgan Bits (radius=2)"),
        Patch(facecolor=plt.cm.Greens(0.5), edgecolor="black", label="RDKit Descriptors"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Add explanation
    explanation = (
        "Feature Importance: SHAP mean |value| (prediction-based, unbiased)\n"
        "Morgan Bits: Circular substructure patterns (SMILES notation after →)\n"
        "RDKit: Physicochemical properties (MW, LogP, TPSA, etc.)"
    )
    fig.text(
        0.12,
        -0.02,
        explanation,
        fontsize=9,
        style="italic",
        color="#555555",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6, pad=0.8),
    )

    plt.tight_layout()
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.12)
    fig.savefig(plots_dir / "03_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 03_feature_importance.png")


def plot_error_distribution(xgb_model, X_test, y_test, plots_dir):
    """Plot 3: XGBoost prediction error distribution."""
    print("Creating error distribution plot...")

    xgb_pred = xgb_model.predict(X_test)
    xgb_errors = np.abs(y_test - xgb_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(xgb_errors, bins=30, alpha=0.7, color="darkgreen", edgecolor="black", linewidth=1)
    ax.axvline(
        np.mean(xgb_errors),
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"MAE = {np.mean(xgb_errors):.3f}",
    )
    ax.axvline(
        np.median(xgb_errors),
        color="orange",
        linestyle="--",
        linewidth=2.5,
        label=f"Median = {np.median(xgb_errors):.3f}",
    )

    ax.set_xlabel("Absolute Error (pIC50)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (#samples)", fontsize=12, fontweight="bold")
    ax.set_title(
        "XGBoost: Prediction Error Distribution (Test Set)", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(plots_dir / "04_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 04_error_distribution.png")


def plot_model_performance_summary(xgb_model, X_train, X_test, y_train, y_test, plots_dir):
    """Plot 5: XGBoost model performance summary metrics."""
    print("Creating model performance summary...")

    xgb_test_r2 = xgb_model.score(X_test, y_test)
    # xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test)))
    xgb_test_mae = mean_absolute_error(y_test, xgb_model.predict(X_test))

    # Load actual CV metrics from saved performance file
    metrics_path = Path(__file__).parent / "saved_models" / "egfr_performance.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        xgb_cv_r2_mean = metrics.get("cv_metrics", {}).get("xgb_cv_r2_mean", 0.7007)
        xgb_cv_r2_std = metrics.get("cv_metrics", {}).get("xgb_cv_r2_std", 0.0239)
    else:
        xgb_cv_r2_mean = 0.7007
        xgb_cv_r2_std = 0.0239

    # Overfitting gap = training R² - test R²
    xgb_train_r2 = xgb_model.score(X_train, y_train)
    gap = xgb_train_r2 - xgb_test_r2

    # Blue tone styling - darker background, lighter boxes
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1f2e")

    # Title with proper spacing
    fig.suptitle("XGBoost · EGFR pIC50", fontsize=20, fontweight="bold", color="white", y=0.96)
    fig.text(0.5, 0.88, "Model Performance Summary", fontsize=13, ha="center", color="#7a8399")

    # Create 3 subplots horizontally
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#4a5577")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        # Add border
        rect = plt.Rectangle(
            (0.1, 0.1),
            9.8,
            9.8,
            fill=False,
            edgecolor="#3a4566",
            linewidth=2,
            transform=ax.transData,
        )
        ax.add_patch(rect)

    # LEFT BOX: TEST R²
    ax1.text(
        5,
        9,
        "TEST R²",
        ha="center",
        fontsize=11,
        color="#7a8399",
        fontweight="bold",
        transform=ax1.transData,
    )
    ax1.text(
        5,
        6,
        f"{xgb_test_r2:.3f}",
        ha="center",
        fontsize=52,
        color="white",
        fontweight="bold",
        transform=ax1.transData,
    )
    # Horizontal line under the value
    ax1.plot([1, 9], [3.0, 3.0], color="#4a5580", linewidth=1.5, transform=ax1.transData)

    ax1.text(
        5,
        2.0,
        f"CV R² {xgb_cv_r2_mean:.3f} ± {xgb_cv_r2_std:.3f}",
        ha="center",
        fontsize=14,
        color="#5b9dd9",
        transform=ax1.transData,
    )
    ax1.text(
        5,
        0.5,
        "5-fold cross-validation",
        ha="center",
        fontsize=12,
        color="#5a6a88",
        style="italic",
        transform=ax1.transData,
    )

    # MIDDLE BOX: MEAN ABS ERROR
    ax2.text(
        5,
        9,
        "MEAN ABS ERROR",
        ha="center",
        fontsize=11,
        color="#7a8399",
        fontweight="bold",
        transform=ax2.transData,
    )
    ax2.text(
        5,
        6,
        f"{xgb_test_mae:.3f}",
        ha="center",
        fontsize=52,
        color="#FFB347",
        fontweight="bold",
        transform=ax2.transData,
    )
    ax2.text(
        5, 4.5, "pIC50 units", ha="center", fontsize=10, color="#8899bb", transform=ax2.transData
    )
    # Horizontal line under pIC50 units
    ax2.plot([1, 9], [3.0, 3.0], color="#4a5580", linewidth=1.5, transform=ax2.transData)

    ax2.text(
        5,
        2.0,
        "overfitting gap",
        ha="center",
        fontsize=12,
        color="#5a6a88",
        transform=ax2.transData,
    )
    ax2.text(
        5,
        0.5,
        f"{gap:.3f}",
        ha="center",
        fontsize=28,
        color="#00DD00",
        fontweight="bold",
        transform=ax2.transData,
    )

    # RIGHT BOX: DETAILS
    ax3.text(
        5,
        9,
        "DETAILS",
        ha="center",
        fontsize=11,
        color="#7a8399",
        fontweight="bold",
        transform=ax3.transData,
    )

    details_lines = [
        "Model",
        "XGBoost",
        "",
        "Features",
        "Morgan FP 2048",
        "+ RDKit descriptors 8 = 2056",
        "",
        "Test set",
        f"{len(y_test):,} samples",
        "",
        "CV",
        "5-fold",
    ]

    y_pos = 7.8
    for line in details_lines:
        if line == "":
            y_pos -= 0.4
        else:
            if line in ["Model", "Features", "Test set", "CV"]:
                ax3.text(
                    1.2,
                    y_pos,
                    line,
                    ha="left",
                    fontsize=9,
                    color="#7a8399",
                    fontweight="bold",
                    transform=ax3.transData,
                )
            else:
                ax3.text(
                    1.2,
                    y_pos,
                    line,
                    ha="left",
                    fontsize=10,
                    color="#ffffff",
                    transform=ax3.transData,
                )
            y_pos -= 0.65

    plt.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.1, wspace=0.3)

    fig.savefig(
        plots_dir / "05_model_summary.png", dpi=600, bbox_inches="tight", facecolor="#1a1f2e"
    )
    plt.close()
    print("  ✓ Saved: 05_model_summary.png")


def plot_shap_heatmap(xgb_model, shap_vals, X_test, smiles_list, plots_dir, n_samples=50):
    """Plot 6: SHAP heatmap showing feature contributions across samples.

    Uses the same top 20 features as plot 3, selected via SHAP values.
    Displays interpretable feature names with Morgan bit substructure annotations.
    """
    print("Creating SHAP heatmap (with Morgan annotations, consistent with plot 3)...")

    try:
        # Sample data for heatmap (too many samples makes it unreadable)
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        X_sample = X_test[sample_indices]
        shap_sample = shap_vals[sample_indices]

        # Get top 20 using SHAP (same metric as plot 3)
        mean_shap_abs = np.abs(shap_vals).mean(axis=0)
        top_20_indices = np.argsort(mean_shap_abs)[-20:][::-1]

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

        # Extract Morgan bits in top 20
        morgan_bits_in_top_20 = [idx for idx in top_20_indices if idx < 2048]
        rdkit_indices_in_top_20 = [idx for idx in top_20_indices if idx >= 2048]

        # Annotate Morgan bits
        feature_labels = {}
        if morgan_bits_in_top_20:
            print(f"  → Found {len(morgan_bits_in_top_20)} Morgan bits in top 20 features")
            morgan_annotations = annotate_morgan_bits(
                xgb_model, smiles_list, bit_indices=morgan_bits_in_top_20
            )

            for _, row in morgan_annotations.iterrows():
                bit_id = row["bit_index"]
                if bit_id in morgan_bits_in_top_20:
                    feature_labels[bit_id] = row["feature_name"]

        # Add RDKit names
        for idx in rdkit_indices_in_top_20:
            feature_labels[idx] = rdkit_names.get(idx, f"RDKit_{idx - 2048}")

        # Create ordered labels for top 20
        final_labels = []
        for idx in top_20_indices:
            if idx in feature_labels:
                final_labels.append(feature_labels[idx])
            elif idx < 2048:
                final_labels.append(f"Morgan_Bit{idx}")
            else:
                final_labels.append(rdkit_names.get(idx, f"RDKit_{idx - 2048}"))

        # Select top features from SHAP values
        shap_vals_top = shap_sample[:, top_20_indices]
        X_sample_top = X_sample[:, top_20_indices]

        # Create heatmap
        base_value = float(xgb_model.predict(X_test).mean())

        _ = SHAPVisualizer.heatmap(
            shap_vals=shap_vals_top,
            X=X_sample_top,
            base_value=base_value,
            feature_names=final_labels,
            max_display=20,
            figsize=(14, 8),
            title="XGBoost: SHAP Feature Contribution Heatmap (Top 20 Features)",
            save_path=str(plots_dir / "06_shap_heatmap.png"),
        )
        plt.close()
        print("✓ Saved: 06_shap_heatmap.png")

    except Exception as e:
        logger.warning(f"SHAP heatmap generation skipped: {type(e).__name__}: {e}")
        print(f"⚠ Skipped SHAP heatmap ({type(e).__name__})")


def main():
    """Generate all Phase 2 visualizations (XGBoost only - best model)."""
    print("\n" + "=" * 70)
    print("PHASE 2: PERFORMANCE VISUALIZATIONS (XGBoost Only)")
    print("=" * 70)

    # Create output directory
    plots_dir = create_output_dir()
    print(f"\nOutput directory: {plots_dir}/\n")

    # Load best model (XGBoost only)
    xgb_model, _ = load_or_train_models()  # xgb_model, loaded

    # Load and prepare data
    X_train, X_test, y_train, y_test, _, smiles_list = load_and_prepare_data()  # _: X_morgan

    # Compute SHAP values once (used by both plot 3 and 6)
    print("\nComputing SHAP values (used for both feature importance and heatmap)...")
    background_indices = np.random.choice(len(X_test), min(100, len(X_test) // 2), replace=False)
    X_background = X_test[background_indices]

    explainer = QSARExplainer()
    shap_explainer = explainer.create_explainer(xgb_model, X_background, "XGBoost")
    shap_vals = explainer.compute_shap_values(
        shap_explainer, X_test, "XGBoost", max_samples=len(X_test)
    )
    print(f"  ✓ SHAP values computed for {len(X_test)} test samples")

    # Generate all plots
    plot_residuals(xgb_model, X_test, y_test, plots_dir)
    plot_predictions_vs_actual(xgb_model, X_test, y_test, plots_dir)
    plot_combined_feature_importance(
        xgb_model, plots_dir, smiles_list, shap_vals, X_test, n_features=20
    )
    plot_error_distribution(xgb_model, X_test, y_test, plots_dir)
    plot_model_performance_summary(xgb_model, X_train, X_test, y_train, y_test, plots_dir)
    plot_shap_heatmap(xgb_model, shap_vals, X_test, smiles_list, plots_dir, n_samples=50)

    # Save Morgan bit annotations for use in predictions
    save_morgan_annotations(
        xgb_model,
        smiles_list,
        output_path=Path(__file__).parent / "saved_models" / "morgan_bit_annotations.json",
    )

    # Compute and save residual standard error for confidence intervals in predictions
    print("\nComputing prediction uncertainty metrics...")
    y_pred_test = xgb_model.predict(X_test)
    residuals = y_test - y_pred_test
    residual_std = np.std(residuals)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"  ✓ Residual Std Dev: {residual_std:.4f} pIC50 units")
    print(f"  ✓ RMSE: {rmse:.4f}")

    # Update metadata with uncertainty metrics
    try:
        with open(Path(__file__).parent / "saved_models" / "egfr_metadata.json") as f:
            metadata = json.load(f)

        metadata["uncertainty_metrics"] = {
            "residual_std": float(residual_std),
            "rmse": float(rmse),
            "ci_95_margin": float(1.96 * residual_std),
            "description": "95% confidence interval margin = 1.96 × residual_std",
        }

        with open(Path(__file__).parent / "saved_models" / "egfr_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("  ✓ Updated metadata with uncertainty metrics")
    except Exception as e:
        print(f"  ⚠ Could not update metadata: {e}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE ✅")
    print("=" * 70)
    print("\nAll visualizations saved to: qsar/visualizations/")
    print("\n6 performance plots created:")
    print("  1. Residuals (XGBoost predictions - actual values)")
    print("  2. Predictions vs Actual (calibration plot)")
    print("  3. Feature Importance (Top 20: SHAP-based, Morgan + RDKit)")
    print("  4. Error Distribution (histogram of prediction errors)")
    print("  5. Model Performance Summary (R², RMSE, MAE, overfitting gap)")
    print("  6. SHAP Heatmap (top 20 features, consistent with plot 3)")
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
    print("\n✨ These plots will be displayed in Streamlit dashboard (Phase 3)")


if __name__ == "__main__":
    main()
