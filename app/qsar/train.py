"""
Train QSAR models (RandomForest and XGBoost).

Handles data splitting, model training, and evaluation metrics.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from app.config import logger


class QSARModelTrainer:
    """Train RandomForest and XGBoost models on EGFR IC50 bioactivity data."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        test_size : float
            Train/test split ratio (default 0.2 = 80/20)
        random_state : int
            Random seed for reproducibility (default 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.metrics = {}

        # Store XGBoost parameters for CV evaluation
        self._xgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,  # Moderate tree depth to balance bias-variance
            "subsample": 0.8,
            "colsample_bytree": 0.8,  # sample 80% of features per tree to reduce overfitting
            "reg_alpha": 0.1,  # L1 regularization (lasso) - prevents overfitting to individual samples
            "reg_lambda": 1.5,  # L2 regularization (ridge) - prevents overfitting to individual features
            "min_child_weight": 5,  # Minimum sum of instance weights needed in child node - prevents overfitting to small data subsets
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """
        Split data into train/test sets with validation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (pIC50)

        Returns
        -------
        dict
            Dictionary with X_train, X_test, y_train, y_test

        Raises
        ------
        ValueError
            If data validation fails
        """
        # Input validation
        if X.shape[0] != y.shape[0]:
            error_msg = f"X and y must have same n_samples. Got {X.shape[0]} vs {y.shape[0]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if X.shape[0] < 10:
            error_msg = f"Need at least 10 samples. Got {X.shape[0]}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Fill NaN values in X (RDKit descriptors can produce NaN for edge-case molecules)
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X = X.copy()
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            filled_count = nan_mask.sum()
            logger.warning(f"Filled {filled_count} NaN values in X with column means")

        # Check y for NaN (should not happen in IC50 data, but validate anyway)
        if np.any(np.isnan(y)):
            error_msg = f"y contains {np.sum(np.isnan(y))} NaN values. Cannot train on missing target values."
            logger.error(error_msg)
            raise ValueError(error_msg)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cross_val: bool = False,
        cv_folds: int = 5,
        n_estimators: int = 100,
    ) -> RandomForestRegressor:
        """
        Train RandomForest model.

        Cross-validation is handled centrally in train_both_models() to avoid
        duplicate CV computations and ensure architectural consistency.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets (pIC50)
        n_estimators : int
            Number of trees (default 100)

        Returns
        -------
        RandomForestRegressor
            Trained model

        Raises
        ------
        Exception
            If training fails
        """
        try:
            logger.info(f"Training RandomForest with {n_estimators} trees")

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True,
                max_depth=10,  # Limit tree depth to reduce overfitting (was unlimited)
                min_samples_leaf=5,  # Prevent single-sample leaves
                max_features=0.3,  # Only see 30% of features per split
            )
            if cross_val:
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2")
                print(f"RF CV mean R2: {scores.mean():.3f}")

            model.fit(X_train, y_train)

            # Log out-of-bag score (internal validation estimate)
            oob_r2 = model.oob_score_
            logger.info(f"RandomForest OOB R²: {oob_r2:.4f}")

            # Note: Cross-validation is handled centrally in train_both_models()
            # to avoid duplicate CV computations and ensure consistency

            self.models["rf"] = model
            logger.info("RandomForest training complete")

            return model

        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")
            raise RuntimeError("RandomForest training failed") from e

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cross_val: bool = False,
        cv_folds: int = 5,
        n_estimators: int = 500,
        early_stopping_rounds: int = 20,
        validation_split: float = 0.2,  # Use 20% of training data for early stopping validation
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model with early stopping.

        Early stopping monitors validation RMSE and stops training when
        it stops improving, preventing overfitting and saving computation.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets (pIC50)
        n_estimators : int
            Maximum boosting rounds (default 500, early stopping may use fewer)
        early_stopping_rounds : int
            Stop if validation RMSE doesn't improve for N rounds (default 20)
        validation_split : float
            Fraction of X_train to use for early stopping validation (default 0.2)

        Returns
        -------
        xgb.XGBRegressor
            Trained model

        Raises
        ------
        Exception
            If training fails
        """
        try:
            logger.info(
                f"Training XGBoost with {n_estimators} max rounds, "
                f"early_stopping_rounds={early_stopping_rounds}"
            )

            # Split for early stopping validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=self.random_state
            )

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization (lasso) - prevents overfitting to individual samples
                reg_lambda=1.5,  # L2 regularization (ridge) - shrinks weights toward zero
                min_child_weight=5,  # Minimum sum of instance weights needed in child node
            )
            # Optional cross-validation
            if cross_val:
                cv_model = xgb.XGBRegressor(
                    n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1
                )

                scores = cross_val_score(cv_model, X_train, y_train, cv=cv_folds, scoring="r2")
                print(f"XGB CV mean R2: {scores.mean():.3f}")
            # Train with early stopping
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            actual_rounds = model.best_iteration + 1
            logger.info(
                f"XGBoost training stopped at round {actual_rounds} (early stopping triggered)"
            )

            # Note: Cross-validation is handled centrally in train_both_models()
            # to avoid duplicate CV computations and ensure consistency

            self.models["xgb"] = model
            logger.info("XGBoost training complete")

            return model

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            raise RuntimeError("XGBoost training failed") from e

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
    ) -> dict[str, float]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        model : Any
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        model_name : str
            Model identifier for logging

        Returns
        -------
        dict
            Metrics: r2, rmse, mae
        """
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        }

        self.metrics[model_name] = metrics

        logger.info(f"{model_name} — R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

        return metrics

    def _compute_xgb_cv_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        xgb_params: dict[str, Any],
        n_splits: int = 5,
    ) -> np.ndarray:
        """
        Compute cross-validation scores for XGBoost with early stopping.

        Manual KFold implementation because XGBoost's early stopping requires
        a validation set in each fold, which is incompatible with sklearn's
        cross_val_score.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        xgb_params : dict
            XGBoost hyperparameters
        n_splits : int
            Number of CV folds (default 5)

        Returns
        -------
        np.ndarray
            R² scores for each fold
        """
        logger.info(f"Computing manual KFold CV for XGBoost ({n_splits} splits)")

        # KFold means we shuffle and split the data into n_splits folds,
        # then train on n-1 folds and validate on the remaining fold,
        # repeating this process n_splits times so that each fold serves as
        # the validation set once. This allows us to get a more robust estimate
        # of model performance across different subsets of the data, especially
        # important for small datasets where a single train/test split may not be representative.
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train = X[train_idx]
            X_fold_val = X[val_idx]
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            # Create model for this fold (without early stopping for simplicity)
            fold_params = xgb_params.copy()
            fold_params.pop("early_stopping_rounds", None)  # Remove early stopping for CV

            xgb_fold = xgb.XGBRegressor(**fold_params)
            xgb_fold.fit(X_fold_train, y_fold_train, verbose=False)

            # Score on validation fold
            y_fold_pred = xgb_fold.predict(X_fold_val)
            score = r2_score(y_fold_val, y_fold_pred)
            cv_scores.append(score)

            logger.debug(f"Fold {fold_idx + 1}/{n_splits}: R²={score:.4f}")

        cv_scores = np.array(cv_scores)
        logger.info(f"XGBoost CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return cv_scores

    def train_both_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        early_stopping_rounds: int = 50,
        cross_val: bool = False,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Train both RandomForest and XGBoost models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values (pIC50)
        cross_val : bool
            Enable cross-validation (default False)

        Returns
        -------
        tuple
            (rf_model, xgb_model, evaluation_results)
        """
        try:
            logger.info("Starting model training pipeline")

            # Split data
            splits = self.prepare_data(X, y)

            # Train models
            rf_model = self.train_random_forest(splits["X_train"], splits["y_train"])
            xgb_model = self.train_xgboost(
                splits["X_train"], splits["y_train"], early_stopping_rounds=early_stopping_rounds
            )

            # Evaluate
            rf_metrics = self.evaluate_model(rf_model, splits["X_test"], splits["y_test"], "rf")
            xgb_metrics = self.evaluate_model(xgb_model, splits["X_test"], splits["y_test"], "xgb")
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            raise RuntimeError("Training both models failed") from e

        # Add cross-validation scores only if requested
        if cross_val:
            X_train = splits["X_train"]
            y_train = splits["y_train"]

            # RF cross-validation (standard approach works fine with RF)
            rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="r2")
            rf_metrics["cv_r2_scores"] = rf_cv_scores.tolist()
            rf_metrics["cv_r2_mean"] = float(rf_cv_scores.mean())
            rf_metrics["cv_r2_std"] = float(rf_cv_scores.std())

            # XGBoost cross-validation (manual KFold because of early stopping)
            # IMPORTANT: Use actual_rounds from trained model to match deployed model
            actual_rounds = xgb_model.best_iteration + 1
            xgb_cv_params = self._xgb_params.copy()
            xgb_cv_params["n_estimators"] = actual_rounds  # Match deployed model, not default 500

            xgb_cv_scores = self._compute_xgb_cv_scores(X_train, y_train, xgb_cv_params)
            xgb_metrics["cv_r2_scores"] = xgb_cv_scores.tolist()
            xgb_metrics["cv_r2_mean"] = float(np.mean(xgb_cv_scores))
            xgb_metrics["cv_r2_std"] = float(np.std(xgb_cv_scores))

        logger.info("Model training complete")

        return (
            rf_model,
            xgb_model,
            {
                "splits": splits,
                "metrics": {
                    "rf": rf_metrics,
                    "xgb": xgb_metrics,
                },
            },
        )

    def save_model(
        self,
        model: Any,
        model_name: str,
        filepath: str | Path,
    ) -> bool:
        """
        Save trained model to disk.

        Parameters
        ----------
        model : Any
            Trained model
        model_name : str
            Model identifier for logging
        filepath : str or Path
            Path to save model pickle

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(
                parents=True,  # creates ALL missing parent folders
                exist_ok=True,
            )  # don't crash if folder already exists

            with open(filepath, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"{model_name} saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save {model_name}: {e}")
            return False

    def load_model(
        self,
        filepath: str | Path,
        model_name: str = "model",
    ) -> Any:
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to saved model pickle
        model_name : str
            Model identifier for logging

        Returns
        -------
        Any
            Loaded model or None if failed
        """
        try:
            filepath = Path(filepath)

            with open(filepath, "rb") as f:
                model = pickle.load(f)

            logger.info(f"{model_name} loaded from {filepath}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            return None
