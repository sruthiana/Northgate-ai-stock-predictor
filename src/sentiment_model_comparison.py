
"""Compare existing core features against core features + news sentiment."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.config import load_config, project_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

TARGET = "Target_Next_Day_Log_Return"

CORE_FEATURES = [
    "Daily_Return",
    "Log_Return",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Std",
    "Volatility_20",
    "Volatility_60",
    "DGS10",
    "DGS3MO",
    "Treasury_Spread_10Y_3M",
    "CPIAUCSL",
    "CPI_YoY",
    "UNRATE",
    "UNRATE_Change",
]

SENTIMENT_FEATURES = [
    "News_Count",
    "Average_Sentiment",
    "Positive_Count",
    "Neutral_Count",
    "Negative_Count",
    "Average_Confidence",
    "Has_News",
]


def load_datasets():
    """Load existing and sentiment datasets."""

    processed_dir = project_path(
        load_config()["paths"]["processed_data"]
    )

    baseline = {
        "train": pd.read_parquet(
            processed_dir / "train.parquet"
        ),
        "validation": pd.read_parquet(
            processed_dir / "validation.parquet"
        ),
        "test": pd.read_parquet(
            processed_dir / "test.parquet"
        ),
    }

    sentiment = {
        "train": pd.read_parquet(
            processed_dir / "train_sentiment.parquet"
        ),
        "validation": pd.read_parquet(
            processed_dir / "validation_sentiment.parquet"
        ),
        "test": pd.read_parquet(
            processed_dir / "test_sentiment.parquet"
        ),
    }

    for datasets in [baseline, sentiment]:
        for name, data in datasets.items():
            data["Date"] = pd.to_datetime(data["Date"])

            data.sort_values(
                ["Date", "Ticker"],
                inplace=True,
            )

            data.reset_index(drop=True, inplace=True)

            logger.info(
                "%s dataset shape: %s",
                name,
                data.shape,
            )

    return baseline, sentiment


def validate_datasets(baseline, sentiment):
    """Confirm chronological splits match."""

    for split in ["train", "validation", "test"]:

        base = baseline[split]
        sent = sentiment[split]

        if len(base) != len(sent):
            raise ValueError(
                f"{split} row counts differ: "
                f"{len(base)} vs {len(sent)}"
            )

        base_keys = base[
            ["Date", "Ticker", TARGET]
        ].reset_index(drop=True)

        sent_keys = sent[
            ["Date", "Ticker", TARGET]
        ].reset_index(drop=True)

        if not base_keys.equals(sent_keys):
            raise ValueError(
                f"{split} baseline and sentiment "
                "rows are not aligned."
            )

    logger.info(
        "Baseline and sentiment datasets are aligned."
    )


def build_preprocessor(numeric_features):

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "ticker",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Ticker"],
            ),
        ],
        remainder="drop",
    )


def build_models(numeric_features):

    models = {}

    models["Linear Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    models["Ridge Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features),
            ),
            (
                "model",
                Ridge(alpha=10.0),
            ),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=5.0,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


def calculate_metrics(actual, predicted):

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
        "Actual_Mean": actual.mean(),
        "Predicted_Mean": predicted.mean(),
        "Predicted_Std": predicted.std(),
    }


def prepare_split(data, features):

    required = [
        "Date",
        "Ticker",
        TARGET,
        *features,
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data.copy()

    for feature in features:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

    data[features] = data[features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Preserve rows, and impute using training medians
    # later. No validation/test medians are learned.
    return data


def run_experiment(
    experiment_name,
    train,
    validation,
    test,
    numeric_features,
):

    logger.info(
        "Starting experiment: %s",
        experiment_name,
    )

    train = prepare_split(
        train,
        numeric_features,
    )

    validation = prepare_split(
        validation,
        numeric_features,
    )

    test = prepare_split(
        test,
        numeric_features,
    )

    # Learn medians from training only.
    medians = train[
        numeric_features
    ].median()

    # Drop features that are entirely missing in training.
    usable_features = [
        feature
        for feature in numeric_features
        if pd.notna(medians[feature])
    ]

    if not usable_features:
        raise ValueError(
            "No usable numeric features."
        )

    train[usable_features] = train[
        usable_features
    ].fillna(
        medians[usable_features]
    )

    validation[usable_features] = validation[
        usable_features
    ].fillna(
        medians[usable_features]
    )

    test[usable_features] = test[
        usable_features
    ].fillna(
        medians[usable_features]
    )

    X_train = train[
        usable_features + ["Ticker"]
    ]

    X_validation = validation[
        usable_features + ["Ticker"]
    ]

    X_test = test[
        usable_features + ["Ticker"]
    ]

    y_train = train[TARGET].to_numpy()

    y_validation = validation[TARGET].to_numpy()

    y_test = test[TARGET].to_numpy()

    models = build_models(
        usable_features
    )

    validation_results = []
    test_results = []
    prediction_records = []

    trained_models = {}

    # Train all models and select using validation only.
    for model_name, model in models.items():

        logger.info(
            "%s | Training %s",
            experiment_name,
            model_name,
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_predictions = model.predict(
            X_validation
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_predictions,
        )

        validation_results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Feature_Count": len(usable_features),
            **validation_metrics,
        })

        trained_models[model_name] = model

        logger.info(
            "%s | %s | validation RMSE=%.8f",
            experiment_name,
            model_name,
            validation_metrics["RMSE"],
        )

    validation_df = pd.DataFrame(
        validation_results
    ).sort_values(
        "RMSE"
    ).reset_index(drop=True)

    # Select model based on validation RMSE.
    best_model_name = validation_df.iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    logger.info(
        "%s | Selected model: %s",
        experiment_name,
        best_model_name,
    )

    # Evaluate every trained model on held-out test data.
    for model_name, model in trained_models.items():

        test_predictions = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            test_predictions,
        )

        test_results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Selected_By_Validation": (
                model_name == best_model_name
            ),
            "Feature_Count": len(usable_features),
            **metrics,
        })

        prediction_frame = test[
            ["Date", "Ticker", TARGET]
        ].copy()

        prediction_frame[
            "Experiment"
        ] = experiment_name

        prediction_frame[
            "Model"
        ] = model_name

        prediction_frame[
            "Predicted_Log_Return"
        ] = test_predictions

        prediction_records.append(
            prediction_frame
        )

    # Zero-return reference.
    zero_predictions = np.zeros(
        len(y_test)
    )

    zero_metrics = calculate_metrics(
        y_test,
        zero_predictions,
    )

    test_results.append({
        "Experiment": experiment_name,
        "Model": "Zero Return Baseline",
        "Selected_By_Validation": False,
        "Feature_Count": 0,
        **zero_metrics,
    })

    # Save only the validation-selected model for this experiment.
    models_dir = project_path(
        load_config()["paths"]["models"]
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = models_dir / (
        f"sentiment_comparison_"
        f"{experiment_name.lower().replace(' ', '_')}"
        f"_best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    metadata_path = models_dir / (
        f"sentiment_comparison_"
        f"{experiment_name.lower().replace(' ', '_')}"
        f"_metadata.joblib"
    )

    joblib.dump(
        {
            "experiment": experiment_name,
            "selected_model": best_model_name,
            "features": usable_features,
            "target": TARGET,
            "selection_metric": "Validation RMSE",
            "training_medians": medians.to_dict(),
            "validation_results": validation_df.to_dict(
                orient="records"
            ),
        },
        metadata_path,
    )

    logger.info(
        "Saved selected model: %s",
        model_path,
    )

    return (
        validation_df,
        pd.DataFrame(test_results),
        prediction_records,
    )


def main():

    logger.info(
        "Starting baseline versus sentiment comparison."
    )

    baseline, sentiment = load_datasets()

    validate_datasets(
        baseline,
        sentiment,
    )

    # Run both experiments on the same date splits.
    baseline_results = run_experiment(
        "Baseline",
        baseline["train"],
        baseline["validation"],
        baseline["test"],
        CORE_FEATURES,
    )

    sentiment_results = run_experiment(
        "With Sentiment",
        sentiment["train"],
        sentiment["validation"],
        sentiment["test"],
        CORE_FEATURES + SENTIMENT_FEATURES,
    )

    baseline_validation = baseline_results[0]

    sentiment_validation = sentiment_results[0]

    baseline_test = baseline_results[1]

    sentiment_test = sentiment_results[1]

    baseline_predictions = baseline_results[2]

    sentiment_predictions = sentiment_results[2]

    validation_comparison = pd.concat(
        [
            baseline_validation,
            sentiment_validation,
        ],
        ignore_index=True,
    )

    test_comparison = pd.concat(
        [
            baseline_test,
            sentiment_test,
        ],
        ignore_index=True,
    )

    all_predictions = pd.concat(
        baseline_predictions + sentiment_predictions,
        ignore_index=True,
    )

    reports_dir = project_path(
        load_config()["paths"]["reports"]
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path = (
        reports_dir
        / "sentiment_comparison_validation.csv"
    )

    test_path = (
        reports_dir
        / "sentiment_comparison_test.csv"
    )

    predictions_path = (
        reports_dir
        / "sentiment_comparison_test_predictions.csv"
    )

    validation_comparison.to_csv(
        validation_path,
        index=False,
    )

    test_comparison.to_csv(
        test_path,
        index=False,
    )

    all_predictions.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 90)
    print("VALIDATION COMPARISON")
    print("=" * 90)

    print(
        validation_comparison[
            [
                "Experiment",
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "Directional_Accuracy",
            ]
        ].sort_values(
            ["Experiment", "RMSE"]
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 90)
    print("HELD-OUT TEST COMPARISON")
    print("=" * 90)

    print(
        test_comparison[
            [
                "Experiment",
                "Model",
                "Selected_By_Validation",
                "MAE",
                "RMSE",
                "R2",
                "Directional_Accuracy",
            ]
        ].sort_values(
            ["Experiment", "RMSE"]
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved comparison reports:")
    print(validation_path)
    print(test_path)
    print(predictions_path)

    logger.info(
        "Baseline versus sentiment comparison completed."
    )


if __name__ == "__main__":
    main()
