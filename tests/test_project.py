"""Pytest test suite for MP144 data cleaning, pipeline, and FastAPI endpoints.
All tests build synthetic data in-memory and do not depend on gitignored CSV files.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge

from src.mp144.data import clean_ownership, clean_dataset
from src.mp144.pipeline import build_pipeline
import app.main
from app.main import app


@pytest.fixture
def synthetic_train_df():
    """Create a minimal synthetic dataframe for testing data cleaning."""
    return pd.DataFrame({
        "record_id": ["R001", "R002", "R003", "R004"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "highway_distance_km": [2.5, 5.0, 10.0, 1.2],
        "nearby_pumps": [3, 5, 8, 2],
        "price_rs": [95.0, 98.0, 102.0, 92.0],
        "trucks_passing": [500, 1200, 800, 1500],
        "temp_c": [25.0, 30.0, 22.0, 28.0],
        "staff": [5, 10, 8, 4],
        "ownership": ["COOPERATIVE", "public sector", "pri.", "coo."],
        "zone": ["Highway", "City", "Industrial", "Highway"],
        "litres_from_tank_dip_next_day": [12000, 15000, 11000, 13000],
        "litres_per_day": [12050, 150200, 11040, 130000],  # R002 and R004 have 10x typos
    })


@pytest.fixture
def client():
    """FastAPI TestClient with 127.0.0.1 to prevent slow DNS lookups on Windows."""
    return TestClient(app, base_url="http://127.0.0.1")


def test_clean_ownership_variants():
    """Verify spelling variants are mapped to exact 3 levels."""
    raw = pd.Series(["COOPERATIVE", "coo.", "public sector", " pub.", "pri.", "private", "Unknown"])
    cleaned = clean_ownership(raw)
    expected = ["cooperative", "cooperative", "public_sector", "public_sector", "private", "private", "unknown"]
    assert list(cleaned) == expected


def test_clean_dataset_typos_and_leaks(synthetic_train_df):
    """Verify target typos are corrected and leaking columns are dropped."""
    clean_df = clean_dataset(synthetic_train_df, is_train=True, typo_threshold=50000)

    # Leaking column dropped
    assert "litres_from_tank_dip_next_day" not in clean_df.columns
    # Target typos fixed (values > 50000 divided by 10)
    assert clean_df.loc[clean_df["record_id"] == "R002", "litres_per_day"].iloc[0] == 15020.0
    assert clean_df.loc[clean_df["record_id"] == "R004", "litres_per_day"].iloc[0] == 13000.0
    # Clean ownership
    assert set(clean_df["ownership"]).issubset({"cooperative", "public_sector", "private"})


def test_pipeline_fit_predict_synthetic():
    """Verify pipeline builds and predicts on synthetic data."""
    numeric_features = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing", "temp_c", "staff"]
    categorical_features = ["ownership", "zone"]

    X = pd.DataFrame({
        "highway_distance_km": [1.0, 5.0, 10.0, 2.0],
        "nearby_pumps": [2, 5, 8, 3],
        "price_rs": [95.0, 99.0, 101.0, 92.0],
        "trucks_passing": [500, 1000, 1500, 800],
        "temp_c": [20.0, 25.0, 30.0, 22.0],
        "staff": [4, 8, 12, 6],
        "ownership": ["cooperative", "public_sector", "private", "cooperative"],
        "zone": ["Highway", "City", "Industrial", "Highway"],
    })
    y = np.array([10000.0, 14000.0, 18000.0, 11000.0])

    pipeline = build_pipeline(numeric_features, categorical_features, Ridge(alpha=1.0))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_api_health(client):
    """Test GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_predict_valid(client):
    """Test POST /predict with valid request payload."""
    # Ensure a working dummy model is loaded for test isolation
    numeric_features = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing", "temp_c", "staff"]
    categorical_features = ["ownership", "zone"]
    X = pd.DataFrame([{
        "highway_distance_km": 5.0,
        "nearby_pumps": 4,
        "price_rs": 98.0,
        "trucks_passing": 1000,
        "temp_c": 25.0,
        "staff": 8,
        "ownership": "cooperative",
        "zone": "City",
    }])
    y = np.array([12500.0])
    dummy_model = build_pipeline(numeric_features, categorical_features, Ridge())
    dummy_model.fit(X, y)
    from app import main
    main._model = dummy_model

    payload = {
        "highway_distance_km": 5.0,
        "nearby_pumps": 4,
        "price_rs": 98.0,
        "trucks_passing": 1000,
        "temp_c": 25.0,
        "staff": 8,
        "ownership": "cooperative",
        "zone": "City",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_api_predict_rejects_pumps_above_20(client):
    """Client Rule 1: nearby_pumps above 20 must be rejected with 422."""
    payload = {
        "highway_distance_km": 5.0,
        "nearby_pumps": 25,  # Exceeds max 20
        "price_rs": 98.0,
        "trucks_passing": 1000,
        "temp_c": 25.0,
        "staff": 8,
        "ownership": "cooperative",
        "zone": "City",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_rejects_price_below_45(client):
    """Client Rule 2: price_rs below 45 must be rejected with 422."""
    payload = {
        "highway_distance_km": 5.0,
        "nearby_pumps": 5,
        "price_rs": 40.0,  # Below min 45
        "trucks_passing": 1000,
        "temp_c": 25.0,
        "staff": 8,
        "ownership": "cooperative",
        "zone": "City",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
