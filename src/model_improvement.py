"""Diagnostics for Northgate AI Stock Predictor test predictions.

Run: python -m src.model_improvement
Directional accuracy is not meaningful for a zero-return forecast; this module
reports it as NaN for all-zero prediction sets and provides a separate nonzero
prediction hit rate.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import load_config, project_path

TARGET = "Target_Next_Day_Log_Return"
PREDICTION = "Predicted_Log_Return"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sign_pred = np.sign(predicted)
    nonzero = sign_pred != 0
    return {
        "Observations": int(len(actual)),
        "Actual_Mean": float(np.mean(actual)),
        "Predicted_Mean": float(np.mean(predicted)),
        "Actual_Std": float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0,
        "Predicted_Std": float(np.std(predicted, ddof=1)) if len(predicted) > 1 else 0.0,
        "Mean_Prediction_Error": float(np.mean(predicted - actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "Directional_Accuracy": float(np.mean(np.sign(actual) == sign_pred)) if nonzero.any() else np.nan,
        "Directional_Accuracy_Nonzero_Predictions": float(np.mean(np.sign(actual[nonzero]) == sign_pred[nonzero])) if nonzero.any() else np.nan,
        "Prediction_Zero_Rate": float(np.mean(predicted == 0)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    reports = project_path(load_config()["paths"]["reports"])
    path = reports / "baseline_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.modeling first.")
    data = pd.read_csv(path)
    required = {"Date", "Ticker", TARGET, PREDICTION}
    if required - set(data.columns):
        raise ValueError(f"Prediction file missing columns: {sorted(required - set(data.columns))}")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data[PREDICTION] = pd.to_numeric(data[PREDICTION], errors="coerce")
    data = data.dropna(subset=[TARGET, PREDICTION]).copy()

    overall_rows = []
    for name, group in [("Saved Model", data), ("Zero Return Baseline", data)]:
        pred = group[PREDICTION].to_numpy() if name == "Saved Model" else np.zeros(len(group))
        overall_rows.append({"Model": name, **metrics(group[TARGET], pred)})
    overall = pd.DataFrame(overall_rows)
    base = overall.loc[overall["Model"] == "Zero Return Baseline"].iloc[0]
    overall["RMSE_Improvement_vs_Zero"] = base["RMSE"] - overall["RMSE"]
    overall["MAE_Improvement_vs_Zero"] = base["MAE"] - overall["MAE"]
    overall.to_csv(reports / "model_improvement_overall.csv", index=False)

    ticker_rows = []
    for ticker, group in data.groupby("Ticker", sort=True):
        actual = group[TARGET].to_numpy()
        pred = group[PREDICTION].to_numpy()
        row = {"Ticker": ticker, **metrics(actual, pred)}
        zero = metrics(actual, np.zeros(len(actual)))
        row["Zero_Baseline_MAE"] = zero["MAE"]
        row["Zero_Baseline_RMSE"] = zero["RMSE"]
        row["RMSE_Improvement_vs_Zero"] = zero["RMSE"] - row["RMSE"]
        ticker_rows.append(row)
    pd.DataFrame(ticker_rows).to_csv(reports / "model_improvement_by_ticker.csv", index=False)

    residuals = data[["Date", "Ticker", TARGET, PREDICTION]].copy()
    residuals["Residual"] = residuals[TARGET] - residuals[PREDICTION]
    residuals["Absolute_Residual"] = residuals["Residual"].abs()
    residuals["Squared_Residual"] = residuals["Residual"] ** 2
    residuals.to_csv(reports / "model_improvement_residuals.csv", index=False)

    print("\nOVERALL TEST COMPARISON")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTEST RESULTS BY TICKER")
    print(pd.DataFrame(ticker_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nRESIDUAL SUMMARY")
    print(residuals[["Residual", "Absolute_Residual", "Squared_Residual"]].describe().to_string())
    logging.info("Saved corrected model diagnostics.")


if __name__ == "__main__":
    main()
