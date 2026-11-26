"""FastAPI service for MP144 Fuel Station Daily Sales prediction."""

import os
import json
import logging
import sys
from typing import Dict, Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.mp144.data import clean_ownership

# Configure structured JSON logging to stdout (as per Student Handbook section 9)
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "prediction"):
            log_obj["prediction"] = record.prediction
        if hasattr(record, "inputs"):
            log_obj["inputs"] = record.inputs
        return json.dumps(log_obj)

logger = logging.getLogger("mp144_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]
logger.propagate = False

app = FastAPI(title="MP144 Fuel Sales Prediction Service", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
_model = None


def get_model():
    """Lazy load champion model from disk."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    highway_distance_km: float = Field(..., description="Distance to highway in km")
    nearby_pumps: int = Field(..., le=20, description="Pumps within 3km, max 20")
    price_rs: float = Field(..., ge=45.0, description="Petrol price in Rs, min 45")
    trucks_passing: int = Field(..., ge=0, description="Trucks passing per day")
    temp_c: float = Field(..., description="Max temperature in deg C")
    staff: int = Field(..., ge=1, description="Staff count on duty")
    ownership: str = Field(..., description="Ownership category: cooperative, public_sector, private")
    zone: str = Field(..., description="Zone: Highway, City, Industrial")


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict litres sold per day."""
    try:
        model = get_model()
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently available.",
        )

    # Clean ownership and prepare dataframe with exact feature columns
    cleaned_ownership = clean_ownership(pd.Series([request.ownership])).iloc[0]

    input_df = pd.DataFrame([
        {
            "highway_distance_km": request.highway_distance_km,
            "nearby_pumps": request.nearby_pumps,
            "price_rs": request.price_rs,
            "trucks_passing": request.trucks_passing,
            "temp_c": request.temp_c,
            "staff": request.staff,
            "ownership": cleaned_ownership,
            "zone": request.zone,
        }
    ])

    try:
        preds = model.predict(input_df)
        pred_value = float(preds[0])
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    # Structured JSON logging without personal info
    log_record = logging.LogRecord(
        name="mp144_api",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="prediction_made",
        args=(),
        exc_info=None,
    )
    log_record.prediction = round(pred_value, 2)
    log_record.inputs = {
        "zone": request.zone,
        "ownership": cleaned_ownership,
        "nearby_pumps": request.nearby_pumps,
    }
    logger.handle(log_record)

    return PredictResponse(prediction=pred_value)
