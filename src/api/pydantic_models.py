"""Pydantic request/response schemas for the credit-risk prediction API.

The request mirrors the **customer-level aggregate features** produced by
``src.data_processing.AggregateFeatures`` (the input the servable inference
pipeline expects). The response returns the model's risk probability and the
binary high-risk decision.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Aggregate features describing a single customer's transaction history."""

    transaction_count: int = Field(..., ge=0, description="Number of transactions.")
    total_amount: float = Field(..., description="Sum of signed transaction amounts.")
    avg_amount: float = Field(..., description="Mean signed transaction amount.")
    std_amount: float = Field(..., ge=0, description="Std dev of transaction amounts.")
    min_amount: float = Field(..., description="Minimum transaction amount.")
    max_amount: float = Field(..., description="Maximum transaction amount.")
    total_value: float = Field(..., ge=0, description="Sum of absolute transaction values.")
    avg_value: float = Field(..., ge=0, description="Mean absolute transaction value.")
    std_value: float = Field(..., ge=0, description="Std dev of absolute values.")
    avg_hour: float = Field(..., ge=0, le=23, description="Average transaction hour.")
    recency_days: int = Field(..., ge=0, description="Days since the last transaction.")
    tenure_days: int = Field(..., ge=0, description="Days between first and last transaction.")
    credit_ratio: float = Field(..., ge=0, le=1, description="Share of credit (refund) txns.")
    ProductCategory: str = Field(..., description="Most frequent product category.")
    ChannelId: str = Field(..., description="Most frequent channel.")
    ProviderId: str = Field(..., description="Most frequent provider.")
    PricingStrategy: str = Field(..., description="Most frequent pricing strategy (as string).")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Risk prediction for a customer."""

    risk_probability: float = Field(..., ge=0, le=1, description="P(high risk).")
    is_high_risk: int = Field(..., ge=0, le=1, description="1 if predicted high-risk else 0.")
