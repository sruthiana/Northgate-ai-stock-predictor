
"""
Northgate AI Stock Predictor
Model Improvement Diagnostics

Purpose:
1. Analyze actual versus predicted returns.
2. Examine prediction bias and variability.
3. Evaluate residual patterns.
4. Compare model performance by ticker.
5. Save diagnostic reports.

This module does not retrain or overwrite existing models.
"""

import logging

import numpy as np
import pandas as pd

from src.config import project_path


# --------------------------------------------------
# 1. CONFIGURE LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. DEFINE FILE PATHS
# --------------------------------------------------

PREDICTIONS_PATH = project_path(
    "reports/baseline_test_predictions.csv"
)

OVERALL_OUTPUT_PATH = project_path(
    "reports/model_improvement_overall.csv"
)

TICKER_OUTPUT_PATH = project_path(
    "reports/model_improvement_by_ticker.csv"
)

RESIDUAL_OUTPUT_PATH = project_path(
    "reports/model_improvement_residuals.csv"
)


# --------------------------------------------------
# 3. LOAD TEST PREDICTIONS
# --------------------------------------------------

def load_predictions():

    logger.info("Loading baseline test predictions.")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "Date",
        "Ticker",
        "Target_Next_Day_Log_Return",
        "Predicted_Log_Return",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Target_Next_Day_Log_Return"] = pd.to_numeric(
        df["Target_Next_Day_Log_Return"],
        errors="coerce",
    )

    df["Predicted_Log_Return"] = pd.to_numeric(
        df["Predicted_Log_Return"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Ticker",
            "Target_Next_Day_Log_Return",
            "Predicted_Log_Return",
        ]
    )

    if df.empty:
        raise ValueError(
            "No valid prediction observations remain."
        )

    if not np.isfinite(
        df[
            [
                "Target_Next_Day_Log_Return",
                "Predicted_Log_Return",
            ]
        ].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite values detected in predictions."
        )

    logger.info(
        "Loaded %s valid prediction observations.",
        len(df),
    )

    return df


# --------------------------------------------------
# 4. CALCULATE MODEL METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)

    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    actual_mean = np.mean(actual)

    total_sum_squares = np.sum(
        (actual - actual_mean) ** 2
    )

    residual_sum_squares = np.sum(errors ** 2)

    if total_sum_squares > 0:

        r2 = 1 - (
            residual_sum_squares
            / total_sum_squares
        )

    else:

        r2 = np.nan

    # Directional accuracy is calculated only when
    # predictions are nonzero.
    #
    # A zero-return prediction has no direction and
    # must not be treated as a correct prediction.

    directional_mask = predicted != 0

    if directional_mask.any():

        directional_accuracy = np.mean(
            np.sign(actual[directional_mask])
            == np.sign(predicted[directional_mask])
        )

    else:

        directional_accuracy = np.nan

    return {
        "Observations": len(actual),
        "Actual_Mean": np.mean(actual),
        "Predicted_Mean": np.mean(predicted),
        "Actual_Std": np.std(actual, ddof=1),
        "Predicted_Std": np.std(
            predicted,
            ddof=1,
        ),
        "Mean_Prediction_Error": np.mean(
            predicted - actual
        ),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
    }


# --------------------------------------------------
# 5. OVERALL MODEL DIAGNOSTICS
# --------------------------------------------------

def overall_diagnostics(df):

    logger.info(
        "Calculating overall model diagnostics."
    )

    actual = df[
        "Target_Next_Day_Log_Return"
    ].to_numpy()

    predicted = df[
        "Predicted_Log_Return"
    ].to_numpy()

    zero_predictions = np.zeros_like(actual)

    linear_metrics = calculate_metrics(
        actual,
        predicted,
    )

    zero_metrics = calculate_metrics(
        actual,
        zero_predictions,
    )

    results = {
        "Observations": linear_metrics["Observations"],
        "Actual_Mean": linear_metrics["Actual_Mean"],
        "Predicted_Mean": linear_metrics["Predicted_Mean"],
        "Actual_Std": linear_metrics["Actual_Std"],
        "Predicted_Std": linear_metrics["Predicted_Std"],
        "Mean_Prediction_Error": linear_metrics[
            "Mean_Prediction_Error"
        ],
        "Linear_MAE": linear_metrics["MAE"],
        "Linear_RMSE": linear_metrics["RMSE"],
        "Linear_R2": linear_metrics["R2"],
        "Linear_Directional_Accuracy": (
            linear_metrics["Directional_Accuracy"]
        ),
        "Zero_Baseline_MAE": zero_metrics["MAE"],
        "Zero_Baseline_RMSE": zero_metrics["RMSE"],
        "Zero_Baseline_R2": zero_metrics["R2"],
        "Zero_Baseline_Directional_Accuracy": np.nan,
    }

    result_df = pd.DataFrame([results])

    result_df.to_csv(
        OVERALL_OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Saved overall diagnostics: %s",
        OVERALL_OUTPUT_PATH,
    )

    return result_df


# --------------------------------------------------
# 6. TICKER-SPECIFIC DIAGNOSTICS
# --------------------------------------------------

def ticker_diagnostics(df):

    logger.info(
        "Calculating ticker-specific diagnostics."
    )

    results = []

    for ticker, group in df.groupby("Ticker"):

        actual = group[
            "Target_Next_Day_Log_Return"
        ].to_numpy()

        predicted = group[
            "Predicted_Log_Return"
        ].to_numpy()

        zero_predictions = np.zeros_like(actual)

        linear_metrics = calculate_metrics(
            actual,
            predicted,
        )

        zero_metrics = calculate_metrics(
            actual,
            zero_predictions,
        )

        results.append({
            "Ticker": ticker,
            "Observations": len(group),
            "Actual_Mean": linear_metrics["Actual_Mean"],
            "Predicted_Mean": linear_metrics[
                "Predicted_Mean"
            ],
            "Actual_Std": linear_metrics["Actual_Std"],
            "Predicted_Std": linear_metrics[
                "Predicted_Std"
            ],
            "Linear_MAE": linear_metrics["MAE"],
            "Zero_Baseline_MAE": zero_metrics["MAE"],
            "Linear_RMSE": linear_metrics["RMSE"],
            "Zero_Baseline_RMSE": zero_metrics["RMSE"],
            "Linear_R2": linear_metrics["R2"],
            "Directional_Accuracy": (
                linear_metrics["Directional_Accuracy"]
            ),
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "Ticker"
    ).reset_index(drop=True)

    result_df.to_csv(
        TICKER_OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Saved ticker diagnostics: %s",
        TICKER_OUTPUT_PATH,
    )

    return result_df


# --------------------------------------------------
# 7. RESIDUAL ANALYSIS
# --------------------------------------------------

def residual_diagnostics(df):

    logger.info(
        "Calculating residual diagnostics."
    )

    result = df.copy()

    actual = result[
        "Target_Next_Day_Log_Return"
    ]

    predicted = result[
        "Predicted_Log_Return"
    ]

    # Residual = actual - predicted

    result["Residual"] = (
        actual - predicted
    )

    result["Absolute_Residual"] = (
        result["Residual"].abs()
    )

    result["Squared_Residual"] = (
        result["Residual"] ** 2
    )

    result["Actual_Direction"] = np.sign(
        actual
    )

    result["Predicted_Direction"] = np.sign(
        predicted
    )

    result["Direction_Match"] = (
        result["Actual_Direction"]
        == result["Predicted_Direction"]
    )

    result.to_csv(
        RESIDUAL_OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Saved residual diagnostics: %s",
        RESIDUAL_OUTPUT_PATH,
    )

    return result


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    logger.info(
        "Starting model improvement diagnostics."
    )

    df = load_predictions()

    overall = overall_diagnostics(df)

    by_ticker = ticker_diagnostics(df)

    residuals = residual_diagnostics(df)

    print("\n" + "=" * 70)

    print("OVERALL MODEL IMPROVEMENT DIAGNOSTICS")

    print("=" * 70)

    print(
        overall.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 70)

    print("DIAGNOSTICS BY TICKER")

    print("=" * 70)

    print(
        by_ticker.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 70)

    print("RESIDUAL SUMMARY")

    print("=" * 70)

    print(
        residuals[
            [
                "Residual",
                "Absolute_Residual",
                "Squared_Residual",
            ]
        ].describe().to_string()
    )

    logger.info(
        "Model improvement diagnostics completed."
    )


if __name__ == "__main__":

    main()
