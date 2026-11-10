"""Leakage-aware model comparison for Northgate AI Stock Predictor.

Run from project root:
    python -m src.modeling

This version:
- Fits imputers/scalers only on training data.
- Encodes ticker identity without ordinal encoding.
- Compares regression models against a zero-return benchmark.
- Reports directional accuracy only for non-zero predictions (plus full sign accuracy).
- Never chooses an ML model unless it improves validation RMSE over zero.
- Saves the selected artifact and validation/test prediction files.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import load_config, project_path

TARGET = "Target_Next_Day_Log_Return"
ID_COLUMNS = {"Date", "Ticker", TARGET}
LEAKAGE_TOKENS = (
    "target", "future", "next_day", "forward_return", "lead_", "label",
    "actual_return", "predicted", "prediction", "signal", "recommendation",
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    # Avoid joblib's physical-core probing warning on some Windows systems.
    import os
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 2))


def load_data():
    processed = project_path(load_config()["paths"]["processed_data"])
    frames = {}
    for split in ("train", "validation", "test"):
        path = processed / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python -m src.modeling_data first.")
        frame = pd.read_parquet(path)
        if "Date" not in frame or "Ticker" not in frame or TARGET not in frame:
            raise ValueError(f"{split} must contain Date, Ticker, and {TARGET}.")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame["Ticker"] = frame["Ticker"].astype(str)
        frames[split] = frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return frames["train"], frames["validation"], frames["test"]


def select_features(train: pd.DataFrame) -> list[str]:
    features = []
    for col in train.columns:
        if col in ID_COLUMNS:
            continue
        name = str(col).lower()
        if any(token in name for token in LEAKAGE_TOKENS):
            logging.warning("Excluding possible leakage column: %s", col)
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            features.append(col)
    if not features:
        raise ValueError("No eligible numeric feature columns.")
    return features


def assemble_features(train, validation, test, feature_columns):
    frames = [train, validation, test]
    ticker_levels = sorted(set().union(*(set(f["Ticker"].astype(str)) for f in frames)))
    ticker_cols = [f"Ticker__{t}" for t in ticker_levels]

    xs, ys, valid_frames = [], [], []
    for split_name, frame in zip(("train", "validation", "test"), frames):
        missing = [c for c in feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{split_name} missing features: {missing}")
        X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(frame[TARGET], errors="coerce")
        keep = y.notna()
        X, y, kept = X.loc[keep].reset_index(drop=True), y.loc[keep], frame.loc[keep].reset_index(drop=True)
        encoded = pd.get_dummies(kept["Ticker"].astype(str), prefix="Ticker", dtype=float)
        encoded = encoded.reindex(columns=ticker_cols, fill_value=0.0).reset_index(drop=True)
        X = pd.concat([X, encoded], axis=1)
        xs.append(X)
        ys.append(y.to_numpy(dtype=float))
        valid_frames.append(kept)
    return xs, ys, valid_frames, list(xs[0].columns)


def build_models(seed: int):
    def pipeline(estimator, scaled=False):
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scaled:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)
    return {
        "Ridge": pipeline(Ridge(alpha=25.0), scaled=True),
        "Random Forest": pipeline(RandomForestRegressor(
            n_estimators=350, max_depth=9, min_samples_leaf=15,
            max_features=0.7, random_state=seed, n_jobs=-1)),
        "Extra Trees": pipeline(ExtraTreesRegressor(
            n_estimators=350, max_depth=10, min_samples_leaf=15,
            max_features=0.7, random_state=seed, n_jobs=-1)),
        "Hist Gradient Boosting": pipeline(HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=12,
            l2_regularization=2.0, random_state=seed)),
        "XGBoost": pipeline(XGBRegressor(
            n_estimators=400, max_depth=2, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
            reg_alpha=0.5, reg_lambda=10.0, objective="reg:squarederror",
            eval_metric="rmse", random_state=seed, n_jobs=-1, verbosity=0)),
    }


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty split.")
    sign_actual = np.sign(actual)
    sign_pred = np.sign(predicted)
    nonzero = sign_pred != 0
    return {
        "Observations": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        # All predictions: zero is counted as no directional call.
        "Directional_Accuracy": float(np.mean(sign_actual == sign_pred)),
        # Directional hit rate only for predictions that actually call up/down.
        "Directional_Accuracy_Nonzero_Predictions": (
            float(np.mean(sign_actual[nonzero] == sign_pred[nonzero])) if nonzero.any() else np.nan
        ),
        "Prediction_Zero_Rate": float(np.mean(predicted == 0)),
        "Mean_Prediction": float(np.mean(predicted)),
        "Mean_Actual": float(np.mean(actual)),
        "Prediction_Std": float(np.std(predicted, ddof=1)) if len(predicted) > 1 else 0.0,
    }


def main():
    configure_logging()
    train, validation, test = load_data()
    features = select_features(train)
    Xs, ys, frames, feature_names = assemble_features(train, validation, test, features)
    X_train, X_val, X_test = Xs
    y_train, y_val, y_test = ys
    train_frame, val_frame, test_frame = frames

    seed = int(load_config()["project"]["random_seed"])
    models = build_models(seed)
    validation_rows, validation_predictions, trained = [], {}, {}

    for name, model in models.items():
        logging.info("Training %s", name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
            model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_val), dtype=float)
        validation_predictions[name] = pred
        trained[name] = model
        validation_rows.append({"Model": name, **calculate_metrics(y_val, pred)})

    zero_val = np.zeros_like(y_val)
    zero_metrics = calculate_metrics(y_val, zero_val)
    validation_rows.append({"Model": "Zero Return Baseline", **zero_metrics})
    validation_df = pd.DataFrame(validation_rows).sort_values("RMSE").reset_index(drop=True)

    ml = validation_df[validation_df["Model"] != "Zero Return Baseline"]
    best_ml = ml.sort_values("RMSE").iloc[0]
    if float(best_ml["RMSE"]) < zero_metrics["RMSE"]:
        selected_name = str(best_ml["Model"])
        selected_model = trained[selected_name]
        test_predictions = np.asarray(selected_model.predict(X_test), dtype=float)
        logging.info("Selected %s; validation RMSE beat zero-return benchmark.", selected_name)
    else:
        selected_name = "Zero Return Baseline"
        selected_model = DummyRegressor(strategy="constant", constant=0.0)
        selected_model.fit(X_train, y_train)
        test_predictions = np.zeros_like(y_test)
        logging.warning("No ML candidate beat zero-return validation RMSE; benchmark retained.")

    test_metrics = calculate_metrics(y_test, test_predictions)
    test_df = pd.DataFrame([
        {"Model": selected_name, **test_metrics},
        {"Model": "Zero Return Baseline", **calculate_metrics(y_test, np.zeros_like(y_test))},
    ])

    config = load_config()
    models_dir = project_path(config["paths"]["models"])
    reports_dir = project_path(config["paths"]["reports"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, models_dir / "best_baseline_model.joblib")
    joblib.dump({
        "model_name": selected_name,
        "target": TARGET,
        "feature_columns": feature_names,
        "selection_metric": "validation RMSE; ML selected only if below zero-return baseline",
        "ticker_encoding": "one-hot",
        "seed": seed,
    }, models_dir / "baseline_model_metadata.joblib")

    validation_df.to_csv(reports_dir / "baseline_validation_results.csv", index=False)
    test_df.to_csv(reports_dir / "baseline_test_results.csv", index=False)
    val_blocks = []
    for name, pred in validation_predictions.items():
        block = val_frame[["Date", "Ticker", TARGET]].copy().reset_index(drop=True)
        block["Model"] = name
        block["Predicted_Log_Return"] = pred
        val_blocks.append(block)
    if val_blocks:
        pd.concat(val_blocks, ignore_index=True).to_csv(
            reports_dir / "improved_validation_predictions.csv", index=False
        )
    test_pred_frame = test_frame[["Date", "Ticker", TARGET]].copy()
    test_pred_frame["Predicted_Log_Return"] = test_predictions
    test_pred_frame.to_csv(reports_dir / "baseline_test_predictions.csv", index=False)

    print("\nVALIDATION MODEL COMPARISON (lower RMSE is better)")
    print(validation_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSelected validation decision: {selected_name}")
    print("\nTEST RESULTS (held-out; evaluated after selection)")
    print(test_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
