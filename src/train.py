"""Model training, evaluation, and experiment tracking for the credit-risk model.

This script trains and compares two classifiers for the ``is_high_risk`` proxy
target, tunes their hyperparameters, logs every run to **MLflow** (parameters,
metrics, and the fitted model artifact), and registers the best model in the
MLflow Model Registry.

Workflow
--------
1. Build the model-ready feature table + proxy target from the raw data
   (reusing :mod:`src.data_processing`, so training and serving share one
   feature definition).
2. Stratified train/test split with a fixed ``random_state`` for reproducibility.
3. For each candidate model (Logistic Regression, Random Forest) run a
   ``GridSearchCV`` hyperparameter search, evaluate on the held-out test set,
   and log the run to MLflow.
4. Select the best run by ROC-AUC and register it in the Model Registry.

Run with::

    python -m src.train

then inspect results with ``mlflow ui``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from src.data_processing import RANDOM_STATE, build_processing_pipeline, build_target

TARGET = "is_high_risk"
EXPERIMENT_NAME = "credit-risk-model"
REGISTERED_MODEL_NAME = "credit-risk-classifier"


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_features_and_target(raw_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Build the customer-level feature matrix X and proxy target y from raw data."""
    raw = pd.read_csv(raw_path)

    features = build_processing_pipeline().fit_transform(raw)
    target = build_target(raw)

    data = features.join(target).dropna(subset=[TARGET])
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified train/test split with a fixed random_state for reproducibility."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(model, X_test, y_test) -> dict[str, float]:
    """Compute the standard imbalanced-classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    # Probability of the positive class for ROC-AUC.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


# -----------------------------------------------------------------------------
# Model candidates + hyperparameter grids
# -----------------------------------------------------------------------------
def get_model_candidates() -> dict[str, dict]:
    """Return candidate estimators paired with their hyperparameter search grids.

    ``class_weight='balanced'`` handles the class imbalance in the proxy target.
    """
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 5],
            },
        },
    }


# -----------------------------------------------------------------------------
# Training + tracking
# -----------------------------------------------------------------------------
def train_and_log(
    name: str,
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    cv: int = 3,
) -> dict:
    """Tune one model, evaluate it, and log the run to MLflow.

    Returns a summary dict with the run id, the fitted best estimator, and the
    test metrics.
    """
    with mlflow.start_run(run_name=name) as run:
        search = GridSearchCV(
            estimator,
            param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        metrics = evaluate(best_model, X_test, y_test)

        mlflow.log_param("model_type", name)
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, name="model")

        print(f"[{name}] best params: {search.best_params_}")
        print(f"[{name}] test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        return {
            "name": name,
            "run_id": run.info.run_id,
            "model": best_model,
            "metrics": metrics,
        }


def run_training(
    raw_path: str | Path = "data/raw/data.csv",
    tracking_uri: str | None = None,
    selection_metric: str = "roc_auc",
) -> dict:
    """Full training workflow: load data, train candidates, register the best model."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_features_and_target(raw_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}  Positive rate: {y.mean():.3f}")

    candidates = get_model_candidates()
    results = [
        train_and_log(
            name, cfg["estimator"], cfg["param_grid"], X_train, y_train, X_test, y_test
        )
        for name, cfg in candidates.items()
    ]

    best = max(results, key=lambda r: r["metrics"][selection_metric])
    print(
        f"\nBest model: {best['name']} "
        f"({selection_metric}={best['metrics'][selection_metric]:.4f})"
    )

    # Register the best run's model in the MLflow Model Registry.
    model_uri = f"runs:/{best['run_id']}/model"
    mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"Registered '{REGISTERED_MODEL_NAME}' from run {best['run_id']}")

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and track credit-risk models.")
    parser.add_argument("--input", default="data/raw/data.csv", help="Raw CSV path.")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (defaults to local ./mlruns).",
    )
    args = parser.parse_args()
    run_training(raw_path=args.input, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
