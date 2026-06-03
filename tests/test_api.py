"""Tests for the FastAPI prediction service.

These use a stub model so they run in CI without a trained artifact or MLflow.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app

VALID_PAYLOAD = {
    "transaction_count": 5,
    "total_amount": 12000.0,
    "avg_amount": 2400.0,
    "std_amount": 1500.0,
    "min_amount": -50.0,
    "max_amount": 5000.0,
    "total_value": 12100.0,
    "avg_value": 2420.0,
    "std_value": 1490.0,
    "avg_hour": 13.5,
    "recency_days": 14,
    "tenure_days": 60,
    "credit_ratio": 0.2,
    "ProductCategory": "airtime",
    "ChannelId": "ChannelId_3",
    "ProviderId": "ProviderId_4",
    "PricingStrategy": "2",
}


class _StubModel:
    """Returns a fixed positive-class probability for any input."""

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]] * len(X))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "model", _StubModel())
    return TestClient(app)


def test_health_check_reports_model_loaded(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_probability_and_label(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_probability"] == pytest.approx(0.8)
    assert body["is_high_risk"] == 1


def test_predict_rejects_invalid_input(client):
    bad = dict(VALID_PAYLOAD, credit_ratio=2.0)  # violates le=1 constraint
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_returns_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    client = TestClient(app)
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503
