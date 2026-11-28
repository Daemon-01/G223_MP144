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

    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(train_clean, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    train_counts, _ = np.histogram(train_clean, bins=bin_edges)
    later_counts, _ = np.histogram(later_clean, bins=bin_edges)

    train_pct = np.maximum(train_counts / len(train_clean), 1e-4)
    later_pct = np.maximum(later_counts / len(later_clean), 1e-4)

    train_pct = train_pct / train_pct.sum()
    later_pct = later_pct / later_pct.sum()

    psi_val = np.sum((later_pct - train_pct) * np.log(later_pct / train_pct))
    return float(psi_val)


def run_stage4(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Execute Stage 4 workflow:
    1. Compute PSI for every numeric column (train vs later).
    2. Compute v1's MAE per zone group on later.csv.
    3. Sort later.csv by date, split 70% train / 30% check.
    4. Retrain v2 on train + first 70% of later.csv.
    5. Compare v1 and v2 on the last 30% check set.
    6. Apply promotion rule: promote if v2 MAE is >= 15% lower than v1 MAE.
    7. Log v2 run to MLflow, register version 2, set @champion alias.
    8. Generate holdout predictions on data/raw/holdout_features.csv.
    9. Update findings.yaml and reports/findings.yaml.
    """
    params = load_params(params_path)
    data_cfg = params["data"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    experiment_name = mlflow_cfg["experiment_name"]
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Load and clean data
    raw_train_path = data_cfg["train_path"]
    df_train = pd.read_csv(raw_train_path)
    df_clean_train = clean_dataset(df_train, is_train=True)

    raw_later_path = data_cfg["later_path"]
    df_later = pd.read_csv(raw_later_path)
    df_clean_later = clean_dataset(df_later, is_train=False)

    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]
    features = data_cfg["features"]
    target_col = data_cfg["target_col"]
    reg_model_name = model_cfg.get("name", "mp144_model")

    # 1. PSI calculation
    psi_results = {}
    for col in numeric_features:
        psi_results[col] = round(compute_psi(df_clean_train[col], df_clean_later[col], num_bins=10), 4)
    most_drifted = max(psi_results, key=psi_results.get)

    # Load v1 model
    v1_model_path = model_cfg.get("model_path", "models/model.joblib")
    v1_model = joblib.load(v1_model_path)

    # 2. MAE by group on later.csv
    mae_by_group = {}
    for zone in ["Highway", "City", "Industrial"]:
        sub_df = df_clean_later[df_clean_later["zone"] == zone]
        preds = v1_model.predict(sub_df[features])
        mae_by_group[zone] = round(float(mean_absolute_error(sub_df[target_col], preds)), 2)
    worst_served_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Split later.csv sorted by date
    df_later_sorted = df_clean_later.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_later_sorted) * 0.70)
    later_train = df_later_sorted.iloc[:split_idx]
    later_check = df_later_sorted.iloc[split_idx:]

    # Combined dataset for v2
    v2_train_df = pd.concat([df_clean_train, later_train], ignore_index=True)
    X_v2_train = v2_train_df[features]
    y_v2_train = v2_train_df[target_col]

    X_check = later_check[features]
    y_check = later_check[target_col]

    # Evaluate v1 on check set
    v1_check_preds = v1_model.predict(X_check)
    mae_v1_on_check = round(float(mean_absolute_error(y_check, v1_check_preds)), 2)

    # Train v2
    v2_regressor = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
    )
    v2_pipeline = build_pipeline(numeric_features, categorical_features, v2_regressor)
    v2_pipeline.fit(X_v2_train, y_v2_train)

    # Evaluate v2 on check set
    v2_check_preds = v2_pipeline.predict(X_check)
    mae_v2_on_check = round(float(mean_absolute_error(y_check, v2_check_preds)), 2)

    # 4. Promotion rule: promote if v2 MAE is at least 15% lower than v1 MAE
    improvement = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
    promote = bool(improvement >= 0.15)
    decision = "promote" if promote else "keep"

    # MLflow logging for v2
    data_sha256 = compute_file_sha256(raw_train_path)
    with mlflow.start_run(run_name="retrain_v2") as run:
        v2_run_id = run.info.run_id
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", "GradientBoostingRegressor_v2")
        mlflow.log_param("retrained", True)
        mlflow.log_metric("mae", mae_v2_on_check)
        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_on_check)
        mlflow.log_metric("improvement_pct", round(improvement * 100, 2))

        mlflow.sklearn.log_model(
            v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    # Register model version 2
    v2_model_uri = f"runs:/{v2_run_id}/model"
    v2_version_obj = mlflow.register_model(v2_model_uri, reg_model_name)
    v2_version = str(v2_version_obj.version)

    # Set champion alias according to decision
    if promote:
        champion_version = v2_version
        champion_model = v2_pipeline
        client.set_registered_model_alias(reg_model_name, "champion", champion_version)
        joblib.dump(champion_model, v1_model_path)
    else:
        champion_version = "1"
        champion_model = v1_model

    # 5. Predict holdout file
    raw_holdout_path = data_cfg["holdout_path"]
    df_holdout = pd.read_csv(raw_holdout_path)
    holdout_clean = clean_dataset(df_holdout, is_train=False)
    holdout_preds = champion_model.predict(holdout_clean[features])

    os.makedirs("predictions", exist_ok=True)
    predictions_path = "predictions/holdout_predictions.csv"
    pred_df = pd.DataFrame({
        "record_id": df_holdout["record_id"],
        "prediction": holdout_preds,
    })
    pred_df.to_csv(predictions_path, index=False)

    # 6. Update findings.yaml
    findings_data = {
        "group_id": "G223",
        "problem_id": "MP144",
        "data_issues": {
            "leaking_column": {
                "present": True,
                "column": "litres_from_tank_dip_next_day",
            },
            "unit_mixup": {
                "present": False,
                "column": None,
                "group_value": None,
                "factor": None,
            },
            "sentinel_value": {
                "present": False,
                "column": None,
                "value": None,
                "count": None,
            },
            "duplicate_rows": {
                "present": False,
                "count": None,
            },
            "stuck_sensor": {
                "present": False,
                "column": None,
                "start_date": None,
                "end_date": None,
            },
            "target_typos": {
                "present": True,
                "count": 11,
            },
            "spelling_variants": {
                "present": True,
                "column": "ownership",
                "true_levels": 3,
            },
        },
        "tracking": {
            "experiment_name": experiment_name,
            "best_run_id": "f981f7d717354dc48443536c845f8fc2",
            "registered_model_name": reg_model_name,
            "champion_version": int(champion_version),
        },
        "monitoring": {
            "psi": psi_results,
            "most_drifted_feature": most_drifted,
            "mae_by_group": mae_by_group,
            "worst_served_group": worst_served_group,
            "mae_v1_on_check": mae_v1_on_check,
            "mae_v2_on_check": mae_v2_on_check,
            "decision": decision,
        },
    }

    # Format findings.yaml cleanly with comments as expected by the marking tool
    yaml_text = f"""# MP144 findings --- fill in every value, keep every key.
# The marking tool reads this file. Write null only where the comment says so.
# Use the exact column names and group values from your data files.

group_id: "G223"          # your group id, e.g. "G042"
problem_id: "MP144"

data_issues:
  # For each of the 7 kinds: present: true or false.
  # If present is false, leave the other values as null.
  leaking_column:
    present: true
    column: litres_from_tank_dip_next_day           # the column that must not be used as a feature
  unit_mixup:
    present: false
    column: null           # the column recorded in a different unit
    group_value: null      # the value of zone whose rows are affected
    factor: null           # wrong value = right value x factor
  sentinel_value:
    present: false
    column: null
    value: null            # the special number typed instead of a blank
    count: null            # how many rows have it
  duplicate_rows:
    present: false
    count: null            # number of EXTRA copies (rows you would delete)
  stuck_sensor:
    present: false
    column: null
    start_date: null       # first date with the stuck value, YYYY-MM-DD
    end_date: null         # last date with the stuck value, YYYY-MM-DD
  target_typos:
    present: true
    count: 11          # rows whose target was typed 10 times too big
  spelling_variants:
    present: true
    column: ownership
    true_levels: 3      # how many real categories there are after cleaning

tracking:
  experiment_name: "{experiment_name}"
  best_run_id: "f981f7d717354dc48443536c845f8fc2"        # the MLflow run id of your v1 model
  registered_model_name: "{reg_model_name}"
  champion_version: {champion_version}

monitoring:
  psi:                     # cleaned train.csv vs later.csv, 10 bins
    highway_distance_km: {psi_results['highway_distance_km']}
    nearby_pumps: {psi_results['nearby_pumps']}
    price_rs: {psi_results['price_rs']}
    trucks_passing: {psi_results['trucks_passing']}
    temp_c: {psi_results['temp_c']}
    staff: {psi_results['staff']}
  most_drifted_feature: "{most_drifted}"
  mae_by_group:            # v1 on later.csv, per value of zone
    'Highway': {mae_by_group['Highway']}
    'City': {mae_by_group['City']}
    'Industrial': {mae_by_group['Industrial']}
  worst_served_group: "{worst_served_group}"
  mae_v1_on_check: {mae_v1_on_check}    # v1 on the last 30% of later.csv (by date)
  mae_v2_on_check: {mae_v2_on_check}    # v2 on the same rows
  decision: "{decision}"           # promote or keep
"""
    with open("findings.yaml", "w") as f:
        f.write(yaml_text)

    os.makedirs("reports", exist_ok=True)
    with open("reports/findings.yaml", "w") as f:
        f.write(yaml_text)

    return findings_data
