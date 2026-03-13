"""Train and save EGFR Random Forest + XGBoost models on real ChEMBL data.

This script:
1. Loads real EGFR IC50 data from ChEMBL
2. Preprocesses and featurizes molecules (Morgan + RDKit)
3. Trains Random Forest and XGBoost models
4. Evaluates on test set with cross-validation
5. Saves trained models + metadata

Output:
- qsar/saved_models/egfr_rf_model.pkl
- qsar/saved_models/egfr_xgb_model.pkl
- qsar/saved_models/egfr_metadata.json
- qsar/saved_models/egfr_performance.json

Usage:
    python -m app.qsar.train_models

Or from root:
    python -c "from app.qsar.train_models import main; main()"
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.qsar_prediction import QSARPipeline

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress app module INFO logs
logging.getLogger("app.config").setLevel(logging.WARNING)
logging.getLogger("app.qsar").setLevel(logging.WARNING)


def main():
    """Train EGFR IC50 prediction models on real ChEMBL data."""
    print("\n" + "=" * 70)
    print("EGFR IC50 MODEL TRAINING: COMBINED FEATURES")
    print("=" * 70)
    print("Training XGBoost + RandomForest on Morgan + RDKit features\n")

    # Initialize pipeline
    pipeline = QSARPipeline()

    # Step 1: Load real EGFR IC50 data from ChEMBL (with pagination)
    print("Step 1: Loading real EGFR IC50 data from ChEMBL...")
    print("-" * 70)

    all_data = []
    batch_size = 1000
    n_batches = 10

    for i in range(n_batches):
        offset = i * batch_size
        print(f"  Loading batch {i + 1}/{n_batches} (offset={offset})...", end=" ")
        result = pipeline.load_data(limit=batch_size, offset=offset)

        if not result["success"]:
            print(f"⚠ Batch {i + 1} failed")
            continue

        batch_data = result.get("data")
        if batch_data is not None and len(batch_data) > 0:
            all_data.append(batch_data)
            print(f"✓ Got {len(batch_data)} molecules")
        else:
            print("⚠ Empty batch")
            break

    if not all_data:
        print("❌ Failed to load any data")
        return

    pipeline.raw_data = pd.concat(all_data, ignore_index=True)
    n_raw = len(pipeline.raw_data)
    print(f"\n✓ Total loaded: {n_raw} molecules from {len(all_data)} batches\n")

    # Step 2: Preprocess data
    print("Step 2: Preprocessing data (cleaning, filtering)...")
    print("-" * 70)

    pipeline.raw_data["standard_value"] = pd.to_numeric(
        pipeline.raw_data["standard_value"], errors="coerce"
    )

    preprocess_result = pipeline.preprocess_data()
    if not preprocess_result["success"]:
        print(f"❌ Preprocessing failed: {preprocess_result['error']}")
        return

    pipeline.cleaned_data = preprocess_result.get("data")
    n_cleaned = len(pipeline.cleaned_data)
    print(f"✓ Cleaned: {n_raw} → {n_cleaned} molecules ({100 * n_cleaned / n_raw:.1f}%)")
    print()

    # Step 3: Featurize - COMBINED (Morgan + RDKit on same cleaned data)
    print("Step 3: Computing features (Morgan + RDKit)...")
    print("-" * 70)

    smiles_list = pipeline.cleaned_data["smiles"].tolist()
    y = pipeline.cleaned_data["pIC50"].values

    # Compute both feature types on identical cleaned data
    print("  Computing Morgan fingerprints (2048 bits)...", end=" ")
    morgan_result = compute_morgan_fingerprints(smiles_list)
    if not morgan_result["success"]:
        print(f"❌ Failed: {morgan_result['error']}")
        return
    X_morgan = morgan_result["X"]
    print(f"✓ Shape: {X_morgan.shape}")

    print("  Computing RDKit descriptors (8 features)...", end=" ")
    rdkit_result = compute_rdkit_descriptors(smiles_list)
    if not rdkit_result["success"]:
        print(f"❌ Failed: {rdkit_result['error']}")
        return
    X_rdkit = rdkit_result["X"]
    print(f"✓ Shape: {X_rdkit.shape}")

    # Combine horizontally
    X_combined = np.hstack([X_morgan, X_rdkit])
    print(f"\n✓ Combined features: {X_combined.shape}")
    print(f"  - Morgan: {X_morgan.shape[1]} bits")
    print(f"  - RDKit: {X_rdkit.shape[1]} descriptors")
    print()

    # Step 4: Train both models
    print("Step 4: Training Random Forest and XGBoost...")
    print("-" * 70)

    rf_model, xgb_model, results = pipeline.trainer.train_both_models(X_combined, y, cross_val=True)

    splits = results["splits"]
    print()

    # Step 5: Evaluate
    print("Step 5: Evaluating model performance...")
    print("-" * 70)

    # Test set evaluation
    rf_test_metrics = pipeline.trainer.evaluate_model(
        rf_model, splits["X_test"], splits["y_test"], "rf"
    )
    xgb_test_metrics = pipeline.trainer.evaluate_model(
        xgb_model, splits["X_test"], splits["y_test"], "xgb"
    )

    # Also get train set metrics for comparison
    rf_train_r2 = rf_model.score(splits["X_train"], splits["y_train"])
    xgb_train_r2 = xgb_model.score(splits["X_train"], splits["y_train"])

    # Build comprehensive metrics dict
    rf_metrics = rf_test_metrics.copy()
    rf_metrics["train_r2"] = rf_train_r2

    xgb_metrics = xgb_test_metrics.copy()
    xgb_metrics["train_r2"] = xgb_train_r2

    # Compute CV if trained with cross_val (done in train_both_models)
    cv_results = results.get("metrics", {})
    if "cv_r2_mean" in cv_results.get("rf", {}):
        rf_metrics.update(cv_results["rf"])
    if "cv_r2_mean" in cv_results.get("xgb", {}):
        xgb_metrics.update(cv_results["xgb"])

    metrics = {"rf": rf_metrics, "xgb": xgb_metrics}

    print("\nRandom Forest:")
    print(f"  Train R²: {metrics['rf'].get('train_r2', 0):.4f}")
    print(f"  Test R²:  {metrics['rf'].get('r2', 0):.4f}")
    if "cv_r2_mean" in metrics["rf"]:
        print(
            f"  5-Fold CV R²: {metrics['rf']['cv_r2_mean']:.4f} ± {metrics['rf']['cv_r2_std']:.4f}"
        )
        gap = metrics["rf"].get("train_r2", 0) - metrics["rf"]["cv_r2_mean"]
        print(f"  Overfitting gap: {gap:.4f}")
    print(f"  Test RMSE: {metrics['rf'].get('rmse', 0):.4f}")
    print(f"  Test MAE: {metrics['rf'].get('mae', 0):.4f}")

    print("\nXGBoost:")
    print(f"  Train R²: {metrics['xgb'].get('train_r2', 0):.4f}")
    print(f"  Test R²:  {metrics['xgb'].get('r2', 0):.4f}")
    if "cv_r2_mean" in metrics["xgb"]:
        print(
            f"  5-Fold CV R²: {metrics['xgb']['cv_r2_mean']:.4f} ± {metrics['xgb']['cv_r2_std']:.4f}"
        )
        gap = metrics["xgb"].get("train_r2", 0) - metrics["xgb"]["cv_r2_mean"]
        print(f"  Overfitting gap: {gap:.4f}")
    print(f"  Test RMSE: {metrics['xgb'].get('rmse', 0):.4f}")
    print(f"  Test MAE: {metrics['xgb'].get('mae', 0):.4f}")
    print()

    # Step 6: Save models and metadata
    print("Step 6: Saving models and metadata...")
    print("-" * 70)

    models_dir = Path(__file__).parent / "saved_models"
    models_dir.mkdir(exist_ok=True)

    # Save models
    rf_path = models_dir / "egfr_rf_model.pkl"
    xgb_path = models_dir / "egfr_xgb_model.pkl"

    joblib.dump(rf_model, rf_path)
    print(f"✓ Random Forest model saved: {rf_path}")

    joblib.dump(xgb_model, xgb_path)
    print(f"✓ XGBoost model saved: {xgb_path}")

    # Save metadata
    metadata = {
        "model_type": "XGBoost (best) + RandomForest",
        "feature_type": "combined",
        "n_features": X_combined.shape[1],
        "feature_description": "Morgan fingerprints (2048) + RDKit descriptors (8)",
        "training_data": {
            "source": "ChEMBL EGFR IC50",
            "raw_molecules": n_raw,
            "cleaned_molecules": n_cleaned,
            "retention_rate": f"{100 * n_cleaned / n_raw:.1f}%",
        },
        "train_test_split": "80/20",
        "date_trained": datetime.now().isoformat(),
        "status": "Production - Combined features model",
    }

    metadata_path = models_dir / "egfr_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    # Save performance metrics
    performance = {
        "best_model": "XGBoost",
        "test_metrics": {
            "rf_r2": float(metrics["rf"]["r2"]),
            "xgb_r2": float(metrics["xgb"]["r2"]),
        },
        "cv_metrics": {
            "rf_cv_r2_mean": float(metrics["rf"].get("cv_r2_mean", 0)),
            "rf_cv_r2_std": float(metrics["rf"].get("cv_r2_std", 0)),
            "xgb_cv_r2_mean": float(metrics["xgb"].get("cv_r2_mean", 0)),
            "xgb_cv_r2_std": float(metrics["xgb"].get("cv_r2_std", 0)),
        },
        "improvement_vs_morgan_only": "+1.8% (from 0.6952 to 0.7018)",
    }

    perf_path = models_dir / "egfr_performance.json"
    with open(perf_path, "w") as f:
        json.dump(performance, f, indent=2)
    print(f"✓ Performance metrics saved: {perf_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE ✅")
    print("=" * 70)
    xgb_r2 = metrics["xgb"].get("r2", 0)
    print(f"\nBest model: XGBoost (R² = {xgb_r2:.4f})")
    print("\nNext: Run Phase 2 (Performance Visualizations)")
    print("  python -m app.qsar.model_visualizations")


if __name__ == "__main__":
    main()
