"""Unit tests for src/data_processing.py."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    AggregateFeatures,
    DateTimeFeatures,
    WOETransformer,
    build_processing_pipeline,
    calculate_woe_iv,
)


@pytest.fixture
def raw_df():
    """Small hand-built transaction frame: 2 customers, known aggregates."""
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C1", "C2"],
            "TransactionStartTime": [
                "2018-11-15T02:18:49Z",
                "2018-11-16T10:00:00Z",
                "2018-11-20T23:30:00Z",
                "2018-12-01T08:00:00Z",
            ],
            "Amount": [100.0, -50.0, 200.0, 1000.0],
            "Value": [100, 50, 200, 1000],
            "ProductCategory": ["airtime", "airtime", "tv", "financial_services"],
            "ChannelId": ["ChannelId_3", "ChannelId_2", "ChannelId_3", "ChannelId_3"],
            "ProviderId": ["ProviderId_4", "ProviderId_4", "ProviderId_6", "ProviderId_1"],
            "PricingStrategy": [2, 2, 4, 2],
        }
    )


def test_datetime_features_extract_calendar_parts(raw_df):
    out = DateTimeFeatures().fit_transform(raw_df)
    for col in ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]:
        assert col in out.columns
    assert out["transaction_hour"].iloc[0] == 2
    assert out["transaction_day"].iloc[0] == 15
    assert out["transaction_month"].iloc[0] == 11
    assert out["transaction_year"].iloc[0] == 2018


def test_aggregate_reduces_to_one_row_per_customer(raw_df):
    dt = DateTimeFeatures().fit_transform(raw_df)
    agg = AggregateFeatures().fit_transform(dt)
    assert agg.shape[0] == 2  # C1, C2
    assert agg.index.name == "CustomerId"
    assert agg.loc["C1", "transaction_count"] == 3
    assert agg.loc["C1", "total_amount"] == pytest.approx(250.0)
    assert agg.loc["C2", "transaction_count"] == 1


def test_aggregate_handles_single_transaction_std(raw_df):
    """A customer with one transaction must get std 0, not NaN."""
    dt = DateTimeFeatures().fit_transform(raw_df)
    agg = AggregateFeatures().fit_transform(dt)
    assert agg.loc["C2", "std_amount"] == 0.0
    assert agg.loc["C2", "std_value"] == 0.0


def test_aggregate_credit_ratio_and_mode(raw_df):
    dt = DateTimeFeatures().fit_transform(raw_df)
    agg = AggregateFeatures().fit_transform(dt)
    # C1 has 1 of 3 transactions negative.
    assert agg.loc["C1", "credit_ratio"] == pytest.approx(1 / 3)
    # Modal category for C1 is 'airtime' (appears twice).
    assert agg.loc["C1", "ProductCategory"] == "airtime"


def test_full_pipeline_is_numeric_scaled_and_complete(raw_df):
    out = build_processing_pipeline().fit_transform(raw_df)
    assert out.shape[0] == 2
    assert not out.isnull().any().any()
    assert (out.dtypes != object).all()  # everything encoded to numeric


def test_calculate_woe_iv_returns_nonnegative_iv():
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    table, iv = calculate_woe_iv(df, "cat", "target")
    assert iv >= 0
    assert "woe" in table.columns


def test_woe_transformer_maps_categories_to_numbers():
    df = pd.DataFrame({"cat": ["a", "a", "b", "b", "a", "b"]})
    y = np.array([0, 0, 1, 1, 0, 1])
    out = WOETransformer(columns=["cat"]).fit_transform(df, y)
    assert pd.api.types.is_numeric_dtype(out["cat"])
    # Unseen category falls back to neutral WoE (0).
    unseen = WOETransformer(columns=["cat"]).fit(df, y).transform(pd.DataFrame({"cat": ["z"]}))
    assert unseen["cat"].iloc[0] == 0.0
