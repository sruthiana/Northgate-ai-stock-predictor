"""Prepare leakage-aware chronological train/validation/test datasets.

Run from project root: python -m src.modeling_data
"""
from __future__ import annotations
import logging
import pandas as pd
from src.config import load_config, project_path

TARGET = "Target_Next_Day_Log_Return"
EXCLUDED = {"Date", "Ticker", TARGET}
LEAKAGE_TOKENS = ("target", "future", "next_day", "forward_return", "lead_", "label", "predicted", "prediction")


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)


def load_combined_data():
    path = project_path(load_config()["paths"]["processed_data"]) / "combined_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Combined dataset not found: {path}")
    data = pd.read_parquet(path)
    required = {"Date", "Ticker", TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Ticker"] = data["Ticker"].astype(str)
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return data


def validate_dataset(data):
    if data["Date"].isna().any():
        raise ValueError("Date contains missing or invalid values.")
    if data["Ticker"].isna().any():
        raise ValueError("Ticker contains missing values.")
    dupes = data.duplicated(["Date", "Ticker"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate Date-Ticker rows.")


def prepare_modeling_dataset(data):
    config = load_config()
    universe = config["data"]["equity_universe"]
    data = data[data["Ticker"].isin(universe)].copy()
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()

    candidates = [c for c in data.columns if c not in EXCLUDED]
    features = []
    for col in candidates:
        if any(token in str(col).lower() for token in LEAKAGE_TOKENS):
            logging.warning("Excluding possible leakage feature: %s", col)
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            features.append(col)
    if not features:
        raise ValueError("No numeric features available after leakage checks.")

    # Keep rows with partially missing features. Train-only median imputation is
    # applied later in modeling.py, preventing whole-date loss and split leakage.
    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data[features] = data[features].replace([float("inf"), float("-inf")], pd.NA)
    data.attrs["feature_columns"] = features
    logging.info("Rows retained (valid target): %d; numeric features: %d", len(data), len(features))
    logging.info("Rows with any missing feature: %d", data[features].isna().any(axis=1).sum())
    return data


def chronological_split(data):
    fraction = float(load_config()["modeling"]["test_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    dates = pd.Index(data["Date"].drop_duplicates().sort_values())
    if len(dates) < 3:
        raise ValueError("Not enough unique dates for chronological splitting")
    test_idx = min(max(int(len(dates) * (1 - fraction)), 1), len(dates) - 1)
    test_start = dates[test_idx]
    dev = data[data["Date"] < test_start].copy()
    test = data[data["Date"] >= test_start].copy()
    dev_dates = pd.Index(dev["Date"].drop_duplicates().sort_values())
    val_idx = min(max(int(len(dev_dates) * .80), 1), len(dev_dates) - 1)
    val_start = dev_dates[val_idx]
    train = dev[dev["Date"] < val_start].copy()
    val = dev[dev["Date"] >= val_start].copy()
    for label, frame in (("train", train), ("validation", val), ("test", test)):
        if frame.empty:
            raise ValueError(f"{label} split is empty")
        logging.info("%s: %s to %s | rows=%d", label, frame.Date.min(), frame.Date.max(), len(frame))
    if train.Date.max() >= val.Date.min() or val.Date.max() >= test.Date.min():
        raise AssertionError("Chronological split overlap detected")
    return train, val, test


def save_modeling_data(train, validation, test):
    output = project_path(load_config()["paths"]["processed_data"])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    logging.info("Saved chronological datasets to %s", output)


def main():
    configure_logging()
    data = load_combined_data()
    validate_dataset(data)
    modeling = prepare_modeling_dataset(data)
    train, validation, test = chronological_split(modeling)
    save_modeling_data(train, validation, test)
    print(f"Prepared {len(modeling):,} rows | features={len(modeling.attrs['feature_columns'])}")
    print(f"Train={train.shape}, Validation={validation.shape}, Test={test.shape}")

if __name__ == "__main__":
    main()
