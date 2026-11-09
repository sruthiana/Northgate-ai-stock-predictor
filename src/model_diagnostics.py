
"""Diagnostic analysis of baseline stock return predictions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import load_config, project_path


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_predictions():
    reports_dir = project_path(
        load_config()["paths"]["reports"]
    )

    path = reports_dir / "baseline_test_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {path}"
        )

    data = pd.read_csv(path)

    data["Date"] = pd.to_datetime(data["Date"])

    required = [
        "Date",
        "Ticker",
        "Target_Next_Day_Log_Return",
        "Predicted_Log_Return",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    return data


def calculate_overall_diagnostics(data):
    actual = data[
        "Target_Next_Day_Log_Return"
    ].to_numpy()

    predicted = data[
        "Predicted_Log_Return"
    ].to_numpy()

    zero_predictions = np.zeros(len(data))

    result = {
        "Observations": len(data),
        "Actual_Mean": actual.mean(),
        "Actual_Std": actual.std(),
        "Predicted_Mean": predicted.mean(),
        "Predicted_Std": predicted.std(),
        "Mean_Prediction_Error": (
            predicted.mean() - actual.mean()
        ),
        "Linear_MAE": mean_absolute_error(
            actual,
            predicted,
        ),
        "Linear_RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "Zero_Baseline_MAE": mean_absolute_error(
            actual,
            zero_predictions,
        ),
        "Zero_Baseline_RMSE": np.sqrt(
            mean_squared_error(
                actual,
                zero_predictions,
            )
        ),
        "Linear_Directional_Accuracy": np.mean(
            np.sign(actual) == np.sign(predicted)
        ),
    }

    return pd.DataFrame([result])


def calculate_ticker_diagnostics(data):
    results = []

    for ticker, group in data.groupby("Ticker"):
        actual = group[
            "Target_Next_Day_Log_Return"
        ].to_numpy()

        predicted = group[
            "Predicted_Log_Return"
        ].to_numpy()

        results.append({
            "Ticker": ticker,
            "Observations": len(group),
            "Actual_Mean": actual.mean(),
            "Predicted_Mean": predicted.mean(),
            "Actual_Std": actual.std(),
            "Predicted_Std": predicted.std(),
            "Linear_MAE": mean_absolute_error(
                actual,
                predicted,
            ),
            "Zero_Baseline_MAE": mean_absolute_error(
                actual,
                np.zeros(len(actual)),
            ),
            "Linear_RMSE": np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            ),
            "Zero_Baseline_RMSE": np.sqrt(
                mean_squared_error(
                    actual,
                    np.zeros(len(actual)),
                )
            ),
            "Directional_Accuracy": np.mean(
                np.sign(actual) == np.sign(predicted)
            ),
        })

    return pd.DataFrame(results)


def save_diagnostics(overall, ticker_results):
    reports_dir = project_path(
        load_config()["paths"]["reports"]
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall.to_csv(
        reports_dir / "model_diagnostics_overall.csv",
        index=False,
    )

    ticker_results.to_csv(
        reports_dir / "model_diagnostics_by_ticker.csv",
        index=False,
    )

    logging.info(
        "Saved diagnostic reports to %s",
        reports_dir,
    )


def main():
    configure_logging()

    logging.info("Starting model diagnostics.")

    data = load_predictions()

    overall = calculate_overall_diagnostics(data)

    ticker_results = calculate_ticker_diagnostics(data)

    save_diagnostics(
        overall,
        ticker_results,
    )

    print("\nOVERALL DIAGNOSTICS")
    print(
        overall.round(6).to_string(index=False)
    )

    print("\nDIAGNOSTICS BY TICKER")
    print(
        ticker_results.round(6).to_string(index=False)
    )

    logging.info("Model diagnostics completed.")


if __name__ == "__main__":
    main()
