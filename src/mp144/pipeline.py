"""Training, MLflow tracking, evaluation, and registry for MP144."""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.mp144.data import clean_dataset, compute_file_sha256


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration from params.yaml with yaml.safe_load."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def build_pipeline(
    numeric_features: list,
    categorical_features: list,
    regressor: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessing and regressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_v1_models(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Train 3+ model settings on cleaned train.csv, log to MLflow, register champion, and save model.joblib."""
    params = load_params(params_path)

    # Setup paths and tracking
    data_cfg = params["data"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    experiment_name = mlflow_cfg["experiment_name"]
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Load and clean training data
    raw_train_path = data_cfg["train_path"]
    train_df = pd.read_csv(raw_train_path)
    clean_df = clean_dataset(
        train_df,
        is_train=True,
        target_col=data_cfg["target_col"],
        typo_threshold=data_cfg.get("target_typo_threshold", 50000),
    )

    # Compute data sha256 (required by Rubric E)
    data_sha256 = compute_file_sha256(raw_train_path)

    features = data_cfg["features"]
    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]
    target_col = data_cfg["target_col"]

    X = clean_df[features]
    y = clean_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=model_cfg.get("test_size", 0.2), random_state=model_cfg.get("random_state", 42)
    )

    # 3 candidate model settings
    candidates = [
        {
            "name": "ridge_v1",
            "model_type": "Ridge",
            "params": {"alpha": 1.0},
            "regressor": Ridge(alpha=1.0),
        },
        {
            "name": "rf_v1",
            "model_type": "RandomForestRegressor",
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        },
        {
            "name": "gbr_v1",
            "model_type": "GradientBoostingRegressor",
            "params": {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 4, "random_state": 42},
            "regressor": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        },
    ]

    best_val_mae = float("inf")
    best_run_id = None
    best_pipeline = None

    for cand in candidates:
        with mlflow.start_run(run_name=cand["name"]) as run:
            run_id = run.info.run_id

            # Log required data_sha256
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", cand["model_type"])
            mlflow.log_params(cand["params"])

            # Build and fit pipeline
            pipeline = build_pipeline(numeric_features, categorical_features, cand["regressor"])
            pipeline.fit(X_train, y_train)

            # Evaluate
            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)
            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            # Metric name contains 'mae' (required by Rubric E)
            mlflow.log_metric("mae", val_mae)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)

            # Log model artifact with serialization_format="cloudpickle"
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run_id
                best_pipeline = pipeline

    # Register best model in MLflow Model Registry
    reg_model_name = model_cfg.get("name", "mp144_model")
    model_uri = f"runs:/{best_run_id}/model"
    model_version_obj = mlflow.register_model(model_uri, reg_model_name)
    champion_version = str(model_version_obj.version)

    # Set champion alias
    client.set_registered_model_alias(reg_model_name, "champion", champion_version)

    # Save champion model to models/model.joblib
    os.makedirs(model_cfg.get("model_dir", "models"), exist_ok=True)
    model_save_path = model_cfg.get("model_path", "models/model.joblib")
    joblib.dump(best_pipeline, model_save_path)

    return {
        "best_run_id": best_run_id,
        "best_val_mae": best_val_mae,
        "registered_model_name": reg_model_name,
        "champion_version": champion_version,
        "model_path": model_save_path,
        "data_sha256": data_sha256,
        "experiment_name": experiment_name,
    }


def compute_psi(train_series: pd.Series, later_series: pd.Series, num_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between train and later distributions."""
    train_clean = train_series.dropna()
    later_clean = later_series.dropna()

    # Define bins based on quantiles or uniform spacing of train
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(train_clean, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    train_counts, _ = np.histogram(train_clean, bins=bin_edges)
    later_counts, _ = np.histogram(later_clean, bins=bin_edges)

    # Smooth zero counts
    train_pct = np.maximum(train_counts / len(train_clean), 1e-4)
    later_pct = np.maximum(later_counts / len(later_clean), 1e-4)

    # Normalize after smoothing
    train_pct = train_pct / train_pct.sum()
    later_pct = later_pct / later_pct.sum()

    psi_val = np.sum((later_pct - train_pct) * np.log(later_pct / train_pct))
    return float(psi_val)
