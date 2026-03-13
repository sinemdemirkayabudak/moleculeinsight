"""
Complete QSAR pipeline: data loading, training, prediction, and explanation.

Integrates data_loader, preprocessing, features, train, predict, and explain modules
to build end-to-end bioactivity prediction models.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import logger
from app.qsar.data_loader import get_egfr_ic50_data
from app.qsar.explain import QSARExplainer
from app.qsar.features import compute_morgan_fingerprints, compute_rdkit_descriptors
from app.qsar.predict import QSARPredictor
from app.qsar.preprocessing import get_cleaned_dataset
from app.qsar.train import QSARModelTrainer


class QSARPipeline:
    """Complete QSAR bioactivity prediction pipeline."""

    def __init__(self):
        """Initialize pipeline components."""
        self.raw_data = None
        self.cleaned_data = None
        self.features = None
        self.feature_names = None
        self.feature_type = None

        self.trainer = QSARModelTrainer()
        self.predictor = QSARPredictor()
        self.explainer = QSARExplainer()

        self.rf_model = None
        self.xgb_model = None
        self.splits = None
        self.metrics = None

    def load_data(
        self,
        limit: int = 10000,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Fetch EGFR IC50 data from ChEMBL.

        Parameters
        ----------
        limit : int
            Number of records to fetch (default 10000)
        offset : int
            Pagination offset (default 0)

        Returns
        -------
        dict
            Result with success status and data
        """
        logger.info("Fetching EGFR IC50 data from ChEMBL")

        result = get_egfr_ic50_data(limit=limit, offset=offset)

        if not result.get("success"):
            logger.error(f"Failed to load data: {result.get('error')}")
            return result

        self.raw_data = result.get("data")
        logger.info(f"Loaded {len(self.raw_data)} raw records")

        return {
            "success": True,
            "count": len(self.raw_data),
            "data": self.raw_data,
        }

    def preprocess_data(
        self,
        min_pic50: float = 3.0,
        max_pic50: float = 12.0,
    ) -> dict[str, Any]:
        """
        Clean and preprocess raw IC50 data.

        Parameters
        ----------
        min_pic50 : float
            Minimum pIC50 threshold (default 3.0)
        max_pic50 : float
            Maximum pIC50 threshold (default 12.0)

        Returns
        -------
        dict
            Result with cleaned data and statistics
        """
        if self.raw_data is None:
            logger.error("No raw data. Call load_data() first.")
            return {"success": False, "error": "No raw data loaded"}

        logger.info("Preprocessing data")

        cleaned_df, stats = get_cleaned_dataset(
            self.raw_data,
            min_pic50=min_pic50,
            max_pic50=max_pic50,
        )

        if cleaned_df is None:
            logger.error("Preprocessing failed")
            return {"success": False, "error": "Preprocessing failed", "stats": stats}

        self.cleaned_data = cleaned_df
        logger.info(f"Preprocessing complete: {len(cleaned_df)} molecules")

        return {
            "success": True,
            "data": cleaned_df,
            "stats": stats,
        }

    def featurize_data(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
    ) -> dict[str, Any]:
        """
        Convert SMILES to molecular features.

        Parameters
        ----------
        fingerprint_type : str
            "morgan" (default) for Morgan fingerprints or "rdkit" for RDKit descriptors
        radius : int
            Radius for Morgan fingerprints (default 2, ignored for rdkit)

        Returns
        -------
        dict
            Result with feature matrix and feature names
        """
        if self.cleaned_data is None:
            logger.error("No cleaned data. Call preprocess_data() first.")
            return {"success": False, "error": "No cleaned data"}

        logger.info(f"Featurizing data with {fingerprint_type} features")

        if fingerprint_type == "morgan":
            result = compute_morgan_fingerprints(
                self.cleaned_data["smiles"].tolist(),
                radius=radius,
            )
        elif fingerprint_type == "rdkit":
            result = compute_rdkit_descriptors(
                self.cleaned_data["smiles"].tolist(),
            )
        else:
            logger.error(f"Unsupported fingerprint type: {fingerprint_type}")
            return {"success": False, "error": f"Unsupported type: {fingerprint_type}"}

        if not result.get("success"):
            logger.error(f"Featurization failed: {result.get('error')}")
            return result

        self.features = result.get("X")
        self.feature_names = result.get("feature_names")
        self.feature_type = fingerprint_type

        logger.info(
            f"Featurization complete: {self.features.shape[0]} molecules, "
            f"{self.features.shape[1]} features"
        )

        return {
            "success": True,
            "X": self.features,
            "feature_names": self.feature_names,
            "shape": self.features.shape,
        }

    def train_models(
        self, early_stopping_rounds: int = 50, cross_val: bool = False
    ) -> dict[str, Any]:
        """
        Train RandomForest and XGBoost models.

        Parameters
        ----------
        early_stopping_rounds : int
            Early stopping patience for XGBoost (default 50)
        cross_val : bool
            Enable cross-validation scoring (default False)

        Returns
        -------
        dict
            Result with trained models and evaluation metrics
        """
        if self.features is None or self.cleaned_data is None:
            logger.error(
                "No features or cleaned data. Run load_data(), preprocess_data(), featurize_data() first."
            )
            return {
                "success": False,
                "error": "Missing features or cleaned data",
            }

        logger.info("Training models")

        # Extract target (pIC50)
        y = self.cleaned_data["pIC50"].values

        # Train both models
        self.rf_model, self.xgb_model, results = self.trainer.train_both_models(
            self.features, y, early_stopping_rounds=early_stopping_rounds, cross_val=cross_val
        )

        self.splits = results["splits"]
        self.metrics = results["metrics"]

        logger.info("Model training complete")

        return {
            "success": True,
            "models": {
                "rf": self.rf_model,
                "xgb": self.xgb_model,
            },
            "metrics": self.metrics,
        }

    def get_predictions(
        self,
        model_type: str = "rf",
    ) -> pd.DataFrame:
        """
        Make predictions with confidence intervals on test set.

        Parameters
        ----------
        model_type : str
            "rf" for RandomForest, "xgb" for XGBoost (default "rf")

        Returns
        -------
        pd.DataFrame
            Predictions with confidence intervals
        """
        if self.splits is None:
            logger.error("No model splits. Run train_models() first.")
            return pd.DataFrame()

        model = self.rf_model if model_type == "rf" else self.xgb_model
        model_name = "RandomForest" if model_type == "rf" else "XGBoost"

        if model is None:
            logger.error(f"{model_name} model not trained")
            return pd.DataFrame()

        logger.info(f"Computing predictions for {model_name}")

        predictions = self.predictor.compute_confidence_intervals(
            model,
            self.splits["X_test"],
            self.splits["y_test"],
            model_name=model_name,
        )

        return predictions

    def get_explanations(
        self,
        model_type: str = "rf",
        top_n: int = 20,
    ) -> dict[str, Any]:
        """
        Generate SHAP-based feature importance.

        Parameters
        ----------
        model_type : str
            "rf" for RandomForest, "xgb" for XGBoost (default "rf")
        top_n : int
            Number of top features to return (default 20)

        Returns
        -------
        dict
            Feature importance and SHAP statistics
        """
        if self.splits is None:
            logger.error("No model splits. Run train_models() first.")
            return {}

        model = self.rf_model if model_type == "rf" else self.xgb_model
        model_name = "RandomForest" if model_type == "rf" else "XGBoost"

        if model is None:
            logger.error(f"{model_name} model not trained")
            return {}

        logger.info(f"Generating SHAP explanations for {model_name}")

        # Create explainer
        explainer = self.explainer.create_explainer(
            model,
            self.splits["X_test"],
            model_name=model_name,
        )

        # Compute SHAP values
        shap_vals = self.explainer.compute_shap_values(
            explainer,
            self.splits["X_test"],
            model_name=model_name,
        )

        # Get feature importance
        importance = self.explainer.get_feature_importance(
            shap_vals,
            feature_names=self.feature_names,
            top_n=top_n,
        )

        # Get summary stats
        stats = self.explainer.summary_stats(
            shap_vals,
            feature_names=self.feature_names,
        )

        logger.info(f"SHAP explanations complete for {model_name}")

        return {
            "importance": importance,
            "stats": stats,
            "shap_values": shap_vals,
            "explainer": explainer,
        }

    def run_full_pipeline(
        self,
        limit: int = 10000,
        offset: int = 0,
        min_pic50: float = 3.0,
        max_pic50: float = 12.0,
    ) -> dict[str, Any]:
        """
        Run complete QSAR pipeline from data loading to explanations.

        Parameters
        ----------
        limit : int
            Records to fetch from ChEMBL (default 10000)
        offset : int
            Pagination offset (default 0)
        min_pic50 : float
            Minimum pIC50 threshold (default 3.0)
        max_pic50 : float
            Maximum pIC50 threshold (default 12.0)

        Returns
        -------
        dict
            Complete results including models, metrics, predictions, and explanations
        """
        logger.info("Starting complete QSAR pipeline")

        # 1. Load data
        load_result = self.load_data(limit=limit, offset=offset)
        if not load_result.get("success"):
            return load_result

        # 2. Preprocess
        preprocess_result = self.preprocess_data(min_pic50=min_pic50, max_pic50=max_pic50)
        if not preprocess_result.get("success"):
            return preprocess_result

        # 3. Featurize
        feature_result = self.featurize_data()
        if not feature_result.get("success"):
            return feature_result

        # 4. Train
        train_result = self.train_models()
        if not train_result.get("success"):
            return train_result

        # 5. Predictions
        rf_preds = self.get_predictions("rf")
        xgb_preds = self.get_predictions("xgb")

        # 6. Explanations
        rf_explain = self.get_explanations("rf")
        xgb_explain = self.get_explanations("xgb")

        logger.info("Complete QSAR pipeline finished")

        return {
            "success": True,
            "data": {
                "raw_count": len(self.raw_data),
                "cleaned_count": len(self.cleaned_data),
                "feature_shape": self.features.shape,
                "train_size": len(self.splits["X_train"]),
                "test_size": len(self.splits["X_test"]),
            },
            "models": {
                "rf": self.rf_model,
                "xgb": self.xgb_model,
            },
            "metrics": self.metrics,
            "predictions": {
                "rf": rf_preds,
                "xgb": xgb_preds,
            },
            "explanations": {
                "rf": rf_explain,
                "xgb": xgb_explain,
            },
        }

    def predict_new_molecules(
        self,
        smiles_list: list[str],
        model_type: str = "rf",
        feature_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Predict on new molecules not in training data.

        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings
        model_type : str
            "rf" for RandomForest, "xgb" for XGBoost (default "rf")
        feature_type : str or None
            "morgan" for Morgan fingerprints or "rdkit" for RDKit descriptors.
            If None, uses feature type from training (default None)

        Returns
        -------
        pd.DataFrame
            Predictions with confidence intervals for new molecules
        """
        logger.info(f"Predicting on {len(smiles_list)} new molecules")

        # Use stored feature_type if not explicitly provided
        ftype = feature_type if feature_type is not None else self.feature_type
        if ftype is None:
            logger.error("No feature type specified. Call featurize_data() or load_models() first.")
            return pd.DataFrame()

        # Featurize new molecules using appropriate method
        if ftype == "morgan":
            result = compute_morgan_fingerprints(smiles_list)
        elif ftype == "rdkit":
            result = compute_rdkit_descriptors(smiles_list)
        else:
            logger.error(f"Unsupported feature type: {ftype}")
            return pd.DataFrame()

        if not result.get("success"):
            logger.error(f"Failed to featurize new molecules: {result.get('error')}")
            return pd.DataFrame()

        X_new = result.get("X")

        # Get model
        model = self.rf_model if model_type == "rf" else self.xgb_model
        if model is None:
            logger.error("Model not trained. Run train_models() first.")
            return pd.DataFrame()

        # Predict
        y_pred = self.predictor.predict(model, X_new)

        # Compute uncertainty
        if self.splits is not None:
            train_residuals = self.splits["y_test"] - self.predictor.predict(
                model, self.splits["X_test"]
            )
            std_error = np.std(train_residuals)
        else:
            std_error = 0.5  # Default fallback

        # Create results
        results = pd.DataFrame(
            {
                "smiles": smiles_list,
                "pIC50_pred": y_pred,
                "ci_lower": y_pred - (1.96 * std_error),
                "ci_upper": y_pred + (1.96 * std_error),
            }
        )

        logger.info(f"Predictions complete for {len(results)} molecules")

        return results

    def save_models(
        self,
        output_dir: str = "models",
    ) -> dict[str, Any]:
        """
        Save trained models and metadata to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save models (default "models")

        Returns
        -------
        dict
            Result with success status and saved file paths
        """
        if self.rf_model is None or self.xgb_model is None:
            logger.error("Models not trained. Run train_models() first.")
            return {"success": False, "error": "Models not trained"}

        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving models to {output_dir}")

            # Save models
            rf_path = output_path / "rf_model.pkl"
            xgb_path = output_path / "xgb_model.pkl"

            joblib.dump(self.rf_model, rf_path)
            joblib.dump(self.xgb_model, xgb_path)

            # Save metadata
            metadata = {
                "feature_names": self.feature_names,
                "feature_type": self.feature_type,
                "metrics": self.metrics,
                "feature_shape": self.features.shape if self.features is not None else None,
            }
            metadata_path = output_path / "metadata.pkl"
            joblib.dump(metadata, metadata_path)

            logger.info(f"Models saved: {rf_path}, {xgb_path}\nMetadata saved: {metadata_path}")

            return {
                "success": True,
                "rf_model": str(rf_path),
                "xgb_model": str(xgb_path),
                "metadata": str(metadata_path),
            }

        except Exception as e:
            logger.exception(f"Error saving models: {e}")
            return {"success": False, "error": str(e)}

    def load_models(
        self,
        model_dir: str = "models",
    ) -> dict[str, Any]:
        """
        Load pre-trained models and metadata from disk.

        Parameters
        ----------
        model_dir : str
            Directory containing saved models (default "models")

        Returns
        -------
        dict
            Result with success status and loaded models
        """
        try:
            model_path = Path(model_dir)

            if not model_path.exists():
                logger.error(f"Model directory not found: {model_dir}")
                return {"success": False, "error": f"Directory not found: {model_dir}"}

            logger.info(f"Loading models from {model_dir}")

            # Load models
            rf_path = model_path / "rf_model.pkl"
            xgb_path = model_path / "xgb_model.pkl"
            metadata_path = model_path / "metadata.pkl"

            if not rf_path.exists() or not xgb_path.exists():
                logger.error("Model files not found in directory")
                return {"success": False, "error": "Model files not found"}

            self.rf_model = joblib.load(rf_path)
            self.xgb_model = joblib.load(xgb_path)

            # Load metadata if exists
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                self.feature_names = metadata.get("feature_names")
                self.feature_type = metadata.get("feature_type", "morgan")
                self.metrics = metadata.get("metrics")

            logger.info(
                f"Models loaded from {rf_path} and {xgb_path}\n"
                f"Features: {len(self.feature_names) if self.feature_names else 'Unknown'}"
            )

            return {
                "success": True,
                "rf_model": str(rf_path),
                "xgb_model": str(xgb_path),
                "metadata": str(metadata_path) if metadata_path.exists() else None,
            }

        except Exception as e:
            logger.exception(f"Error loading models: {e}")
            return {"success": False, "error": str(e)}
