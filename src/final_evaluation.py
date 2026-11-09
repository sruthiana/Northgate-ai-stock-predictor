import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = ROOT_DIR / "data" / "processed" / "test.parquet"

MODEL_PATH = ROOT_DIR / "models" / "improved_best_model.joblib"

OUTPUT_DIR = ROOT_DIR / "reports"
PREDICTION_DIR = ROOT_DIR / "data" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TARGET = "Target_Next_Day_Log_Return"

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

RANDOM_STATE = 42


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid observations available for evaluation.")

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    # Directional accuracy:
    # Compare actual and predicted signs.
    # Zero actual returns are excluded.
    nonzero_actual = actual != 0

    if nonzero_actual.sum() > 0:
        directional_accuracy = np.mean(
            np.sign(actual[nonzero_actual])
            == np.sign(predicted[nonzero_actual])
        )
    else:
        directional_accuracy = np.nan

    # Zero-return baseline
    zero_predictions = np.zeros_like(actual)

    zero_mae = mean_absolute_error(
        actual,
        zero_predictions,
    )

    zero_rmse = np.sqrt(
        mean_squared_error(
            actual,
            zero_predictions,
        )
    )

    return {
        "Observations": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
        "Zero_Baseline_MAE": zero_mae,
        "Zero_Baseline_RMSE": zero_rmse,
        "Actual_Mean": actual.mean(),
        "Actual_Std": actual.std(),
        "Predicted_Mean": predicted.mean(),
        "Predicted_Std": predicted.std(),
    }


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

def load_test_data():
    logger.info("Loading held-out test dataset.")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    test_df = pd.read_parquet(TEST_PATH)

    logger.info(
        "Test dataset loaded: %s rows.",
        len(test_df),
    )

    required_columns = (
        FEATURES
        + [
            TARGET,
            "Ticker",
            "Date",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    test_df["Date"] = pd.to_datetime(
        test_df["Date"]
    )

    test_df = test_df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    if test_df["Ticker"].isna().any():
        raise ValueError(
            "Test dataset contains missing ticker values."
        )

    if test_df.duplicated(
        subset=["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker records found."
        )

    if test_df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    logger.info(
        "Test date range: %s to %s",
        test_df["Date"].min(),
        test_df["Date"].max(),
    )

    logger.info(
        "Number of test tickers: %s",
        test_df["Ticker"].nunique(),
    )

    return test_df


# --------------------------------------------------
# LOAD SELECTED MODEL
# --------------------------------------------------

def load_model():
    logger.info("Loading selected Ridge model.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logger.info(
        "Selected model loaded successfully."
    )

    return model


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(test_df):
    logger.info("Preparing test features.")

    X_test = test_df[
        FEATURES + ["Ticker"]
    ].copy()

    # Numeric conversion.
    # The fitted pipeline performs its own
    # preprocessing and imputation.
    for column in FEATURES:
        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce",
        )

    # Validate that no feature is entirely missing.
    entirely_missing = [
        column
        for column in FEATURES
        if X_test[column].isna().all()
    ]

    if entirely_missing:
        raise ValueError(
            "Entirely missing test features: "
            f"{entirely_missing}"
        )

    logger.info(
        "Test feature shape: %s",
        X_test.shape,
    )

    return X_test


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

def generate_predictions(model, X_test):
    logger.info("Generating test predictions.")

    predictions = model.predict(X_test)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(predictions) != len(X_test):
        raise ValueError(
            "Prediction count does not match test rows."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    logger.info(
        "Prediction generation completed."
    )

    return predictions


# --------------------------------------------------
# OVERALL TEST EVALUATION
# --------------------------------------------------

def evaluate_overall(test_df, predictions):
    logger.info("Calculating overall test metrics.")

    actual = test_df[TARGET].to_numpy()

    metrics = calculate_metrics(
        actual,
        predictions,
    )

    results = pd.DataFrame(
        [metrics]
    )

    output_path = (
        OUTPUT_DIR
        / "final_test_evaluation.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved overall test results: %s",
        output_path,
    )

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return results


# --------------------------------------------------
# PER-TICKER EVALUATION
# --------------------------------------------------

def evaluate_by_ticker(test_df, predictions):
    logger.info("Calculating per-ticker test metrics.")

    evaluation_df = test_df[
        ["Date", "Ticker", TARGET]
    ].copy()

    evaluation_df["Predicted_Return"] = predictions

    rows = []

    for ticker, group in evaluation_df.groupby(
        "Ticker"
    ):
        metrics = calculate_metrics(
            group[TARGET].to_numpy(),
            group["Predicted_Return"].to_numpy(),
        )

        metrics["Ticker"] = ticker

        rows.append(metrics)

    results = pd.DataFrame(rows)

    results = results[
        [
            "Ticker",
            "Observations",
            "MAE",
            "RMSE",
            "R2",
            "Directional_Accuracy",
            "Zero_Baseline_MAE",
            "Zero_Baseline_RMSE",
            "Actual_Mean",
            "Actual_Std",
            "Predicted_Mean",
            "Predicted_Std",
        ]
    ]

    results = results.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    output_path = (
        OUTPUT_DIR
        / "final_test_metrics_by_ticker.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved ticker-level results: %s",
        output_path,
    )

    print("\n" + "=" * 80)
    print("TEST METRICS BY TICKER")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return results


# --------------------------------------------------
# SAVE TEST PREDICTIONS
# --------------------------------------------------

def save_predictions(test_df, predictions):
    logger.info("Saving test predictions.")

    prediction_df = test_df[
        ["Date", "Ticker", TARGET]
    ].copy()

    prediction_df["Predicted_Return"] = predictions

    prediction_df["Prediction_Error"] = (
        prediction_df[TARGET]
        - prediction_df["Predicted_Return"]
    )

    prediction_df["Actual_Direction"] = np.sign(
        prediction_df[TARGET]
    )

    prediction_df["Predicted_Direction"] = np.sign(
        prediction_df["Predicted_Return"]
    )

    output_path = (
        PREDICTION_DIR
        / "ridge_test_predictions.parquet"
    )

    prediction_df.to_parquet(
        output_path,
        index=False,
    )

    csv_path = (
        OUTPUT_DIR
        / "ridge_test_predictions.csv"
    )

    prediction_df.to_csv(
        csv_path,
        index=False,
    )

    logger.info(
        "Saved test predictions: %s",
        output_path,
    )

    logger.info(
        "Saved prediction CSV: %s",
        csv_path,
    )

    return prediction_df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    logger.info(
        "Starting final test evaluation."
    )

    test_df = load_test_data()

    model = load_model()

    X_test = prepare_features(
        test_df
    )

    predictions = generate_predictions(
        model,
        X_test,
    )

    overall_results = evaluate_overall(
        test_df,
        predictions,
    )

    ticker_results = evaluate_by_ticker(
        test_df,
        predictions,
    )

    prediction_df = save_predictions(
        test_df,
        predictions,
    )

    print("\n" + "=" * 80)
    print("FINAL EVALUATION COMPLETED")
    print("=" * 80)

    print(
        f"Test observations: {len(test_df)}"
    )

    print(
        f"Tickers evaluated: {test_df['Ticker'].nunique()}"
    )

    print(
        f"Prediction rows saved: {len(prediction_df)}"
    )

    print(
        f"Overall report: "
        f"{OUTPUT_DIR / 'final_test_evaluation.csv'}"
    )

    print(
        f"Ticker report: "
        f"{OUTPUT_DIR / 'final_test_metrics_by_ticker.csv'}"
    )

    print(
        f"Predictions: "
        f"{PREDICTION_DIR / 'ridge_test_predictions.parquet'}"
    )

    logger.info(
        "Final test evaluation completed."
    )


if __name__ == "__main__":
    main()