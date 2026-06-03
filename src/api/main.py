"""FastAPI application for serving credit-risk predictions.

The service loads the trained inference pipeline (preprocessing + model) and
exposes a ``/predict`` endpoint that turns a customer's aggregate features into
a risk probability and a binary high-risk decision.

Model loading order
-------------------
1. If ``MODEL_URI`` is set (e.g. ``models:/credit-risk-classifier/latest``),
   load from the MLflow Model Registry.
2. Otherwise load the local artifact written by training
   (``models/credit_risk_pipeline.joblib``).

MLflow is imported lazily so the API runs even where mlflow is not installed,
as long as the local artifact exists.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CustomerFeatures, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/credit_risk_pipeline.joblib"))
MODEL_URI = os.getenv("MODEL_URI")  # e.g. "models:/credit-risk-classifier/latest"

# Module-level model handle, populated on startup.
model = None


def load_model():
    """Load the inference pipeline from the MLflow registry or the local artifact."""
    if MODEL_URI:
        import mlflow.sklearn  # lazy import: only needed for registry loads

        return mlflow.sklearn.load_model(MODEL_URI)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and keep it for the app's lifetime."""
    global model
    model = load_model()
    yield


app = FastAPI(
    title="Credit Risk Model API",
    description="Predict customer credit-risk probability from behavioral features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check() -> dict:
    """Health probe reporting whether the model is loaded."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Predict the credit-risk probability for a single customer."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model or set MODEL_URI / MODEL_PATH.",
        )

    row = pd.DataFrame([features.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(
        risk_probability=probability,
        is_high_risk=int(probability >= 0.5),
    )
