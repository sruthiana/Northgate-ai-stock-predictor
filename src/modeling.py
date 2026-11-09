
"""Baseline machine learning models for Northgate AI Stock Predictor."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.config import load_config, project_path


TARGET = "Target_Next_Day_Log_Return"

EXCLUDED_COLUMNS = [
    "Date",
    "Ticker",
    TARGET,
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_modeling_data():
    """Load train, validation, and test datasets."""

    processed_dir = project_path(
        load_config()["paths"]["processed_data"]
    )

    train = pd.read_parquet(
        processed_dir / "train.parquet"
    )

    validation = pd.read_parquet(
        processed_dir / "validation.parquet"
    )

    test = pd.read_parquet(
        processed_dir / "test.parquet"
    )

    logging.info("Training data: %s", train.shape)
    logging.info("Validation data: %s", validation.shape)
    logging.info("Test data: %s", test.shape)

    return train, validation, test


def prepare_features(train, validation, test):
    """Separate feature matrices and targets."""

    feature_columns = [
        column
        for column in train.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if TARGET not in train.columns:
        raise ValueError(f"Target column missing: {TARGET}")

    if TARGET in feature_columns:
        raise ValueError("Target leakage detected.")

    X_train = train[feature_columns].astype(float)
    y_train = train[TARGET].astype(float)

    X_validation = validation[feature_columns].astype(float)
    y_validation = validation[TARGET].astype(float)

    X_test = test[feature_columns].astype(float)
    y_test = test[TARGET].astype(float)

    if not all(
        np.isfinite(matrix.to_numpy()).all()
        for matrix in [
            X_train,
            X_validation,
            X_test,
        ]
    ):
        raise ValueError(
            "Non-finite values found in model features."
        )

    logging.info(
        "Model feature count: %s",
        len(feature_columns),
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    )


def build_models(random_seed: int):
    """Create baseline regression models."""

    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_seed,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }

    return models


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate regression and directional metrics."""

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted,
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),

        "R2": r2_score(
            actual,
            predicted,
        ),

        "Directional_Accuracy": np.mean(
            np.sign(actual) == np.sign(predicted)
        ),
    }


def train_and_validate(
    models,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train models and compare validation performance."""

    results = []
    trained_models = {}

    for name, model in models.items():

        logging.info("Training model: %s", name)

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        results.append({
            "Model": name,
            **metrics,
        })

        trained_models[name] = model

        logging.info(
            "%s validation RMSE: %.8f",
            name,
            metrics["RMSE"],
        )

    # Zero-return benchmark.
    zero_predictions = np.zeros(
        len(y_validation)
    )

    zero_metrics = calculate_metrics(
        y_validation,
        zero_predictions,
    )

    results.append({
        "Model": "Zero Return Baseline",
        **zero_metrics,
    })

    results_df = pd.DataFrame(results)

    # Select only among trained ML models.
    ml_results = results_df[
        results_df["Model"] != "Zero Return Baseline"
    ]

    best_model_name = ml_results.sort_values(
        "RMSE"
    ).iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    logging.info(
        "Best validation model: %s",
        best_model_name,
    )

    return (
        results_df,
        best_model_name,
        best_model,
    )


def evaluate_test(
    best_model,
    best_model_name,
    X_test,
    y_test,
):
    """Evaluate the selected model on the held-out test dataset."""

    logging.info(
        "Evaluating selected model on test data: %s",
        best_model_name,
    )

    predictions = best_model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    zero_metrics = calculate_metrics(
        y_test,
        np.zeros(len(y_test)),
    )

    results = pd.DataFrame([
        {
            "Model": best_model_name,
            **metrics,
        },
        {
            "Model": "Zero Return Baseline",
            **zero_metrics,
        },
    ])

    return results, predictions


def save_results(
    validation_results,
    test_results,
    best_model,
    best_model_name,
    feature_columns,
    test_data,
    test_predictions,
):
    """Save model, metrics, feature names, and test predictions."""

    config = load_config()

    models_dir = project_path(
        config["paths"]["models"]
    )

    reports_dir = project_path(
        config["paths"]["reports"]
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save selected model.
    joblib.dump(
        best_model,
        models_dir / "best_baseline_model.joblib",
    )

    # Save metadata for reproducibility.
    metadata = {
        "model_name": best_model_name,
        "target": TARGET,
        "feature_columns": feature_columns,
        "random_seed": config["project"]["random_seed"],
    }

    joblib.dump(
        metadata,
        models_dir / "baseline_model_metadata.joblib",
    )

    validation_results.to_csv(
        reports_dir / "baseline_validation_results.csv",
        index=False,
    )

    test_results.to_csv(
        reports_dir / "baseline_test_results.csv",
        index=False,
    )

    predictions_df = test_data[
        ["Date", "Ticker", TARGET]
    ].copy()

    predictions_df["Predicted_Log_Return"] = (
        test_predictions
    )

    predictions_df.to_csv(
        reports_dir / "baseline_test_predictions.csv",
        index=False,
    )

    logging.info("Saved baseline model and results.")


def main():
    configure_logging()

    logging.info(
        "Starting baseline machine learning."
    )

    train, validation, test = load_modeling_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        train,
        validation,
        test,
    )

    config = load_config()

    models = build_models(
        config["project"]["random_seed"]
    )

    (
        validation_results,
        best_model_name,
        best_model,
    ) = train_and_validate(
        models,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    test_results, test_predictions = evaluate_test(
        best_model,
        best_model_name,
        X_test,
        y_test,
    )

    save_results(
        validation_results,
        test_results,
        best_model,
        best_model_name,
        feature_columns,
        test,
        test_predictions,
    )

    print("\nVALIDATION RESULTS")
    print(
        validation_results.sort_values(
            "RMSE"
        ).to_string(index=False)
    )

    print("\nSELECTED MODEL")
    print(best_model_name)

    print("\nFINAL TEST RESULTS")
    print(
        test_results.to_string(
            index=False
        )
    )

    logging.info(
        "Baseline machine learning completed."
    )


if __name__ == "__main__":
    main()
