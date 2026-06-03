"""Feature engineering for the credit-risk model.

This module turns the raw, transaction-level Xente data into a **model-ready,
customer-level** DataFrame using a single, reproducible
``sklearn.pipeline.Pipeline``.

Pipeline stages
---------------
1. ``DateTimeFeatures``  — extract hour/day/month/year from the timestamp.
2. ``AggregateFeatures`` — aggregate transactions to the customer level
   (RFM + amount statistics + modal categoricals).
3. ``ColumnTransformer`` — impute, scale numericals, one-hot encode
   categoricals, emitting a pandas DataFrame.

Credit risk is a property of the *customer*, not of a single transaction, so the
pipeline aggregates to ``CustomerId``. The resulting feature table is what the
models (Task 5) consume.

Proxy target (RFM)
------------------
The raw data has no default label, so an ``is_high_risk`` proxy is engineered
from behaviour: per-customer Recency/Frequency/Monetary values are scaled and
clustered with K-Means, and the least-engaged cluster (high recency, low
frequency, low monetary) is labelled high-risk. ``build_target`` returns the
label and :func:`process` merges it onto the feature table.

Weight of Evidence / Information Value
--------------------------------------
WoE encoding is a *supervised* transform (it needs the target), and the target
is created in Task 4. It is therefore provided here as a separate, composable
transformer (:class:`WOETransformer`) plus a :func:`calculate_woe_iv` helper,
rather than baked into the default unsupervised pipeline. Once the proxy target
exists, ``build_woe_pipeline`` chains the feature pipeline with the WoE step.

The module is deterministic: a fixed ``RANDOM_STATE`` and stateless transforms
mean the same input always yields the same output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Raw-column configuration -----------------------------------------------------
CUSTOMER_ID = "CustomerId"
TIME_COL = "TransactionStartTime"
AMOUNT_COL = "Amount"
VALUE_COL = "Value"

# Low-cardinality categoricals whose per-customer mode is informative.
CATEGORICAL_COLS = ["ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"]

# Constant / redundant raw columns dropped before aggregation.
DROP_COLS = ["CurrencyCode", "CountryCode"]


# -----------------------------------------------------------------------------
# Custom transformers
# -----------------------------------------------------------------------------
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extract calendar features from a timestamp column.

    Adds ``transaction_hour``, ``transaction_day``, ``transaction_month`` and
    ``transaction_year``. The original timestamp is retained for the downstream
    recency/tenure calculation in :class:`AggregateFeatures`.
    """

    def __init__(self, time_col: str = TIME_COL):
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ts = pd.to_datetime(X[self.time_col], utc=True, errors="coerce")
        X[self.time_col] = ts
        X["transaction_hour"] = ts.dt.hour
        X["transaction_day"] = ts.dt.day
        X["transaction_month"] = ts.dt.month
        X["transaction_year"] = ts.dt.year
        return X


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Aggregate transaction rows to one row per customer.

    Produces:

    * **Monetary** — total, mean, std, min, max of ``Amount`` and ``Value``.
    * **Frequency** — number of transactions.
    * **Recency / tenure** — days since the customer's last transaction
      (relative to a snapshot date) and the span between first and last.
    * **Behavioral** — share of credit (negative-amount) transactions and the
      average transaction hour.
    * **Categorical** — the modal (most frequent) value of each categorical
      column per customer.

    The snapshot date is learned in :meth:`fit` (max timestamp + 1 day) so that
    ``transform`` is reproducible and independent of "today".
    """

    def __init__(
        self,
        customer_id: str = CUSTOMER_ID,
        time_col: str = TIME_COL,
        amount_col: str = AMOUNT_COL,
        value_col: str = VALUE_COL,
        categorical_cols: list[str] | None = None,
    ):
        self.customer_id = customer_id
        self.time_col = time_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y=None):
        ts = pd.to_datetime(X[self.time_col], utc=True, errors="coerce")
        # Snapshot one day after the last observed transaction => recency >= 1.
        self.snapshot_date_ = ts.max() + pd.Timedelta(days=1)
        self.categorical_cols_ = (
            self.categorical_cols if self.categorical_cols is not None else CATEGORICAL_COLS
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ts = pd.to_datetime(X[self.time_col], utc=True, errors="coerce")
        X[self.time_col] = ts

        grouped = X.groupby(self.customer_id)

        agg = grouped.agg(
            transaction_count=(self.amount_col, "count"),
            total_amount=(self.amount_col, "sum"),
            avg_amount=(self.amount_col, "mean"),
            std_amount=(self.amount_col, "std"),
            min_amount=(self.amount_col, "min"),
            max_amount=(self.amount_col, "max"),
            total_value=(self.value_col, "sum"),
            avg_value=(self.value_col, "mean"),
            std_value=(self.value_col, "std"),
            first_txn=(self.time_col, "min"),
            last_txn=(self.time_col, "max"),
            avg_hour=("transaction_hour", "mean"),
        )

        # RFM-style temporal features.
        agg["recency_days"] = (self.snapshot_date_ - agg["last_txn"]).dt.days
        agg["tenure_days"] = (agg["last_txn"] - agg["first_txn"]).dt.days
        agg = agg.drop(columns=["first_txn", "last_txn"])

        # Behavioral: share of credit (refund) transactions per customer.
        credit_ratio = grouped[self.amount_col].apply(lambda s: (s < 0).mean())
        agg["credit_ratio"] = credit_ratio

        # Single-transaction customers have undefined std -> 0 variability.
        agg[["std_amount", "std_value"]] = agg[["std_amount", "std_value"]].fillna(0.0)

        # Modal categorical value per customer.
        for col in self.categorical_cols_:
            if col in X.columns:
                mode = grouped[col].agg(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                )
                agg[col] = mode.astype("object")

        agg.index.name = self.customer_id
        return agg


# -----------------------------------------------------------------------------
# Weight of Evidence / Information Value
# -----------------------------------------------------------------------------
def calculate_woe_iv(
    df: pd.DataFrame, feature: str, target: str, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Compute the WoE table and Information Value for a single feature.

    Numeric features are binned into ``n_bins`` quantile buckets; categorical
    features are used as-is. WoE for a bin is ``ln(%good / %bad)`` where *good*
    is ``target == 0`` and *bad* is ``target == 1``. IV is the sum over bins of
    ``(%good - %bad) * WoE``.

    Returns the per-bin WoE table and the total IV.
    """
    data = df[[feature, target]].copy()

    if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique() > n_bins:
        data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")
    else:
        data["bin"] = data[feature].astype("object")

    grouped = data.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    # Laplace smoothing avoids division by zero / log(0) in empty bins.
    grouped["pct_good"] = (grouped["good"] + 0.5) / (total_good + 0.5)
    grouped["pct_bad"] = (grouped["bad"] + 0.5) / (total_bad + 0.5)
    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    iv = float(grouped["iv"].sum())
    return grouped.reset_index(), iv


class WOETransformer(BaseEstimator, TransformerMixin):
    """Supervised Weight-of-Evidence encoder for categorical/binned features.

    For each configured column it learns a category -> WoE mapping from the
    target in :meth:`fit` and replaces the category with its WoE in
    :meth:`transform`. Unseen categories map to 0 (neutral). The learned IV per
    feature is stored in :attr:`iv_`, a common basis for feature selection.

    Mirrors the behaviour of the ``xverse`` / ``woe`` packages while keeping the
    implementation dependency-free and fully reproducible.
    """

    def __init__(self, columns: list[str] | None = None, n_bins: int = 10):
        self.columns = columns
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y is None:
            raise ValueError("WOETransformer requires a target y.")
        X = X.copy()
        target_name = "_woe_target_"
        X[target_name] = np.asarray(y)

        self.columns_ = (
            self.columns
            if self.columns is not None
            else X.select_dtypes(include="object").columns.tolist()
        )
        self.woe_maps_: dict[str, dict] = {}
        self.iv_: dict[str, float] = {}

        for col in self.columns_:
            woe_table, iv = calculate_woe_iv(X, col, target_name, self.n_bins)
            self.woe_maps_[col] = dict(zip(woe_table["bin"].astype(str), woe_table["woe"]))
            self.iv_[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            mapping = self.woe_maps_[col]
            X[col] = X[col].astype(str).map(mapping).fillna(0.0)
        return X


# -----------------------------------------------------------------------------
# Proxy target: RFM segmentation + high-risk labelling
# -----------------------------------------------------------------------------
def compute_rfm(
    raw: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Compute Recency, Frequency and Monetary values per customer.

    * **Recency** — days between the customer's last transaction and the
      snapshot date (lower = more recently active).
    * **Frequency** — number of transactions.
    * **Monetary** — total absolute transaction value (uses ``Value`` so credits
      and debits do not cancel out).

    The snapshot date defaults to one day after the last transaction in the data,
    making recency reproducible and independent of the current date.
    """
    df = raw.copy()
    ts = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df[TIME_COL] = ts
    if snapshot_date is None:
        snapshot_date = ts.max() + pd.Timedelta(days=1)

    rfm = df.groupby(CUSTOMER_ID).agg(
        recency=(TIME_COL, lambda s: (snapshot_date - s.max()).days),
        frequency=(TIME_COL, "count"),
        monetary=(VALUE_COL, "sum"),
    )
    return rfm


def assign_high_risk_label(
    rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, int]:
    """Cluster customers on scaled RFM and flag the least-engaged segment.

    RFM features are standardised, then segmented with K-Means (fixed
    ``random_state`` for reproducibility). The high-risk cluster is the one whose
    centroid shows **high recency, low frequency and low monetary** value — the
    disengaged customers used as a default proxy.

    Returns the RFM table augmented with ``cluster`` and ``is_high_risk`` columns,
    plus the index of the chosen high-risk cluster.
    """
    rfm = rfm.copy()
    features = rfm[["recency", "frequency", "monetary"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    # Rank clusters by a disengagement score built from centroids in RFM space.
    centroids = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    z = (centroids - centroids.mean()) / centroids.std(ddof=0).replace(0, 1)
    # High recency raises risk; high frequency / monetary lower it.
    risk_score = z["recency"] - z["frequency"] - z["monetary"]
    high_risk_cluster = int(risk_score.idxmax())

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm, high_risk_cluster


def build_target(
    raw: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.Series:
    """Convenience wrapper: raw transactions -> ``is_high_risk`` Series.

    The returned Series is indexed by ``CustomerId`` so it joins directly onto
    the feature table produced by :func:`build_processing_pipeline`.
    """
    rfm = compute_rfm(raw, snapshot_date=snapshot_date)
    labeled, _ = assign_high_risk_label(rfm)
    return labeled["is_high_risk"]


# -----------------------------------------------------------------------------
# Pipeline construction
# -----------------------------------------------------------------------------
def build_column_transformer() -> ColumnTransformer:
    """Impute + scale numericals and impute + one-hot encode categoricals.

    Column selection is dtype-based so it adapts to whatever
    :class:`AggregateFeatures` emits.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_include="object")),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer


def build_processing_pipeline() -> Pipeline:
    """Full unsupervised feature pipeline: raw transactions -> model-ready frame.

    Returns a single ``Pipeline`` whose ``fit_transform`` produces a
    customer-indexed, fully numeric, scaled and encoded ``DataFrame``.
    """
    return Pipeline(
        steps=[
            ("datetime", DateTimeFeatures()),
            ("aggregate", AggregateFeatures()),
            ("transform", build_column_transformer()),
        ]
    )


def build_woe_pipeline(woe_columns: list[str] | None = None) -> Pipeline:
    """Supervised variant: aggregate features then WoE-encode categoricals.

    Use this once the proxy target (Task 4) is available. The WoE step is fitted
    with ``pipeline.fit(X, y)``. Numeric features still pass through imputation
    and scaling.
    """
    return Pipeline(
        steps=[
            ("datetime", DateTimeFeatures()),
            ("aggregate", AggregateFeatures()),
            ("woe", WOETransformer(columns=woe_columns)),
            ("transform", build_column_transformer()),
        ]
    )


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def process(
    input_path: str | Path, output_path: str | Path, with_target: bool = True
) -> pd.DataFrame:
    """Read raw CSV, build features (+ proxy target), and write the model-ready CSV.

    When ``with_target`` is True, the RFM-based ``is_high_risk`` proxy label is
    computed and merged onto the feature table by ``CustomerId``.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    raw = pd.read_csv(input_path)

    pipeline = build_processing_pipeline()
    features = pipeline.fit_transform(raw)

    if with_target:
        target = build_target(raw)
        features = features.join(target)  # aligned on the CustomerId index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit-risk feature engineering.")
    parser.add_argument("--input", default="data/raw/data.csv", help="Raw CSV path.")
    parser.add_argument(
        "--output",
        default="data/processed/features.csv",
        help="Where to write the model-ready feature table.",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Skip computing the is_high_risk proxy target.",
    )
    args = parser.parse_args()

    features = process(args.input, args.output, with_target=not args.no_target)
    cols = features.shape[1]
    msg = f"Wrote {features.shape[0]:,} customers x {cols} columns -> {args.output}"
    if "is_high_risk" in features.columns:
        rate = features["is_high_risk"].mean() * 100
        msg += f"  (high-risk: {features['is_high_risk'].sum():,} / {rate:.1f}%)"
    print(msg)


if __name__ == "__main__":
    main()
