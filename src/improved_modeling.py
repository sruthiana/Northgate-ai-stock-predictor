"""
Northgate AI Stock Predictor
Improved Modeling

Models:
1. Linear Regression
2. Ridge Regression
3. Random Forest
4. XGBoost

Uses:
- Training data for fitting
- Validation data for model selection
- No test data

Outputs:
- reports/improved_model_validation_results.csv
- models/improved_model_metadata.joblib
- models/improved_best_model.joblib
"""

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

from src.config import project_path


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


TRAIN_PATH = project_path(
    "data/processed/train.parquet"
)

VALIDATION_PATH = project_path(
    "data/processed/validation.parquet"
)

RESULTS_PATH = project_path(
    "reports/improved_model_validation_results.csv"
)

MODEL_PATH = project_path(
    "models/improved_best_model.joblib"
)

METADATA_PATH = project_path(
    "models/improved_model_metadata.joblib"
)


TARGET = "Target_Next_Day_Log_Return"


# --------------------------------------------------
# 2. FEATURE CONFIGURATION
# --------------------------------------------------

# Exclude raw price levels because they vary substantially
# between stocks and can dominate pooled modeling.
#
# Use return, momentum, volatility, technical, volume,
# and macroeconomic features.

FEATURES = [
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


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

def load_data(path, name):

    logger.info(
        "Loading %s dataset.",
        name,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    required = [
        "Date",
        "Ticker",
        TARGET,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Ticker",
            TARGET,
        ]
    )

    logger.info(
        "%s dataset: %s rows.",
        name,
        len(df),
    )

    return df


# --------------------------------------------------
# 4. VALIDATE FEATURES
# --------------------------------------------------

def validate_features(train, validation):

    missing_train = [
        feature
        for feature in FEATURES
        if feature not in train.columns
    ]

    missing_validation = [
        feature
        for feature in FEATURES
        if feature not in validation.columns
    ]

    if missing_train:

        raise ValueError(
            f"Training missing features: {missing_train}"
        )

    if missing_validation:

        raise ValueError(
            f"Validation missing features: "
            f"{missing_validation}"
        )

    # Confirm training and validation dates do not overlap.

    train_dates = set(
        train["Date"].unique()
    )

    validation_dates = set(
        validation["Date"].unique()
    )

    overlap = train_dates.intersection(
        validation_dates
    )

    if overlap:

        raise ValueError(
            "Training and validation dates overlap."
        )

    logger.info(
        "Feature validation completed."
    )

    logger.info(
        "Training date range: %s to %s",
        train["Date"].min(),
        train["Date"].max(),
    )

    logger.info(
        "Validation date range: %s to %s",
        validation["Date"].min(),
        validation["Date"].max(),
    )


# --------------------------------------------------
# 5. PREPARE FEATURES
# --------------------------------------------------

def prepare_data(train, validation):

    logger.info(
        "Preparing model features."
    )

    train = train.copy()

    validation = validation.copy()

    # Numeric features.

    numeric_features = FEATURES.copy()

    for feature in numeric_features:

        train[feature] = pd.to_numeric(
            train[feature],
            errors="coerce",
        )

        validation[feature] = pd.to_numeric(
            validation[feature],
            errors="coerce",
        )

    # Missing values are handled using training medians.
    # Median values are learned only from training data.

    medians = train[
        numeric_features
    ].median()

    train[numeric_features] = (
        train[numeric_features]
        .fillna(medians)
    )

    validation[numeric_features] = (
        validation[numeric_features]
        .fillna(medians)
    )

    # Add ticker identity as a categorical feature.

    train["Ticker"] = train[
        "Ticker"
    ].astype(str)

    validation["Ticker"] = validation[
        "Ticker"
    ].astype(str)

    categorical_features = [
        "Ticker"
    ]

    X_train = train[
        numeric_features + categorical_features
    ]

    X_validation = validation[
        numeric_features + categorical_features
    ]

    y_train = train[TARGET].to_numpy()

    y_validation = validation[TARGET].to_numpy()

    logger.info(
        "Training feature shape: %s",
        X_train.shape,
    )

    logger.info(
        "Validation feature shape: %s",
        X_validation.shape,
    )

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        numeric_features,
        categorical_features,
        medians,
    )


# --------------------------------------------------
# 6. PREPROCESSOR
# --------------------------------------------------

def build_preprocessor(
    numeric_features,
    categorical_features,
):

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# --------------------------------------------------
# 7. DEFINE MODELS
# --------------------------------------------------

def build_models(
    numeric_features,
    categorical_features,
):

    models = {}

    # Linear Regression

    models["Linear Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    # Ridge Regression

    models["Ridge Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                Ridge(
                    alpha=10.0,
                ),
            ),
        ]
    )

    # Random Forest

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
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

    # XGBoost

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
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


# --------------------------------------------------
# 8. EVALUATE MODELS
# --------------------------------------------------

def evaluate_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    r2 = r2_score(
        y_validation,
        predictions,
    )

    # Directional accuracy excludes zero predictions.

    nonzero = predictions != 0

    if nonzero.any():

        directional_accuracy = np.mean(
            np.sign(
                y_validation[nonzero]
            )
            == np.sign(
                predictions[nonzero]
            )
        )

    else:

        directional_accuracy = np.nan

    # Compare against zero-return baseline.

    zero_predictions = np.zeros_like(
        y_validation
    )

    zero_mae = mean_absolute_error(
        y_validation,
        zero_predictions,
    )

    zero_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            zero_predictions,
        )
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
        "Zero_Baseline_MAE": zero_mae,
        "Zero_Baseline_RMSE": zero_rmse,
        "Predicted_Mean": predictions.mean(),
        "Predicted_Std": predictions.std(),
    }


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

def main():

    logger.info(
        "Starting improved model training."
    )

    train = load_data(
        TRAIN_PATH,
        "training",
    )

    validation = load_data(
        VALIDATION_PATH,
        "validation",
    )

    validate_features(
        train,
        validation,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        numeric_features,
        categorical_features,
        medians,
    ) = prepare_data(
        train,
        validation,
    )

    models = build_models(
        numeric_features,
        categorical_features,
    )

    results = []

    trained_models = {}

    # --------------------------------------------------
    # TRAIN AND EVALUATE
    # --------------------------------------------------

    for model_name, model in models.items():

        logger.info(
            "Training %s.",
            model_name,
        )

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        metrics["Model"] = model_name

        results.append(
            metrics
        )

        trained_models[
            model_name
        ] = model

        logger.info(
            "%s | MAE=%.6f | RMSE=%.6f | R2=%.6f",
            model_name,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"],
        )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "Directional_Accuracy",
            "Zero_Baseline_MAE",
            "Zero_Baseline_RMSE",
            "Predicted_Mean",
            "Predicted_Std",
        ]
    ]

    results_df = results_df.sort_values(
        "RMSE"
    ).reset_index(
        drop=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    logger.info(
        "Saved validation results: %s",
        RESULTS_PATH,
    )

    # --------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------

    best_model_name = results_df.iloc[0][
        "Model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    logger.info(
        "Best model by validation RMSE: %s",
        best_model_name,
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    metadata = {
        "best_model_name": best_model_name,
        "target": TARGET,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "selection_metric": "Validation RMSE",
        "validation_results": results_df.to_dict(
            orient="records"
        ),
        "training_feature_medians": medians.to_dict(),
    }

    joblib.dump(
        metadata,
        METADATA_PATH,
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print("\n" + "=" * 80)

    print("IMPROVED MODEL VALIDATION RESULTS")

    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print("\n" + "=" * 80)

    print(
        f"BEST MODEL: {best_model_name}"
    )

    print("=" * 80)

    print(
        f"Saved model: {MODEL_PATH}"
    )

    print(
        f"Saved metadata: {METADATA_PATH}"
    )

    logger.info(
        "Improved modeling completed."
    )


if __name__ == "__main__":

    main()