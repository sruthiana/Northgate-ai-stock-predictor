
"""Prepare chronological datasets for machine learning."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config, project_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_combined_data() -> pd.DataFrame:
    """Load the combined market and macroeconomic dataset."""

    config = load_config()

    data_path = (
        project_path(config["paths"]["processed_data"])
        / "combined_features.parquet"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Combined dataset not found: {data_path}"
        )

    data = pd.read_parquet(data_path)

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    logging.info("Loaded dataset: %s", data.shape)

    return data


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the dataset before preparing model inputs."""

    required_columns = [
        "Date",
        "Ticker",
        "Target_Next_Day_Log_Return",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data["Date"].isna().any():
        raise ValueError("Date contains missing values.")

    if data["Ticker"].isna().any():
        raise ValueError("Ticker contains missing values.")

    duplicates = data.duplicated(
        subset=["Date", "Ticker"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate Date-Ticker rows."
        )

    logging.info("Dataset validation passed.")


def prepare_modeling_dataset(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Select usable rows and exclude identifiers and target from features."""

    config = load_config()

    equities = config["data"]["equity_universe"]

    data = data[
        data["Ticker"].isin(equities)
    ].copy()

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    # Exclude rows without a valid next-day target.
    data = data.dropna(
        subset=["Target_Next_Day_Log_Return"]
    )

    # Exclude identifiers and target from model feature columns.
    excluded_columns = {
        "Date",
        "Ticker",
        "Target_Next_Day_Log_Return",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    # Keep only numeric model features.
    numeric_features = data[
        feature_columns
    ].select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_features:
        raise ValueError(
            "No numeric features available for modeling."
        )

    # Replace infinite values with missing values.
    data[numeric_features] = data[
        numeric_features
    ].replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    logging.info(
        "Selected %s numeric features.",
        len(numeric_features),
    )

    logging.info(
        "Rows before feature missing-value filtering: %s",
        len(data),
    )

    # For the initial modeling dataset, retain rows with complete features.
    data = data.dropna(
        subset=numeric_features
    ).copy()

    logging.info(
        "Rows after complete-feature filtering: %s",
        len(data),
    )

    # Store feature names in dataframe attributes for downstream use.
    data.attrs["feature_columns"] = numeric_features

    return data


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into chronological train, validation, and test sets."""

    config = load_config()

    test_fraction = config["modeling"]["test_fraction"]

    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    unique_dates = sorted(
        data["Date"].drop_duplicates()
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "Not enough unique dates for chronological splitting."
        )

    # First reserve the final portion of dates for testing.
    test_start_index = int(
        len(unique_dates) * (1 - test_fraction)
    )

    test_start_index = min(
        max(test_start_index, 1),
        len(unique_dates) - 1,
    )

    test_start_date = unique_dates[
        test_start_index
    ]

    development_data = data[
        data["Date"] < test_start_date
    ].copy()

    test_data = data[
        data["Date"] >= test_start_date
    ].copy()

    # Use the final 20% of development dates for validation.
    development_dates = sorted(
        development_data["Date"].unique()
    )

    validation_start_index = int(
        len(development_dates) * 0.80
    )

    validation_start_index = min(
        max(validation_start_index, 1),
        len(development_dates) - 1,
    )

    validation_start_date = development_dates[
        validation_start_index
    ]

    train_data = development_data[
        development_data["Date"] < validation_start_date
    ].copy()

    validation_data = development_data[
        development_data["Date"] >= validation_start_date
    ].copy()

    logging.info(
        "Training dates: %s to %s",
        train_data["Date"].min(),
        train_data["Date"].max(),
    )

    logging.info(
        "Validation dates: %s to %s",
        validation_data["Date"].min(),
        validation_data["Date"].max(),
    )

    logging.info(
        "Testing dates: %s to %s",
        test_data["Date"].min(),
        test_data["Date"].max(),
    )

    return train_data, validation_data, test_data


def save_modeling_data(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save the chronological modeling datasets."""

    config = load_config()

    output_dir = project_path(
        config["paths"]["processed_data"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_parquet(
        output_dir / "train.parquet",
        index=False,
    )

    validation_data.to_parquet(
        output_dir / "validation.parquet",
        index=False,
    )

    test_data.to_parquet(
        output_dir / "test.parquet",
        index=False,
    )

    logging.info(
        "Saved training, validation, and test datasets."
    )


def main() -> None:
    configure_logging()

    logging.info(
        "Starting modeling dataset preparation."
    )

    data = load_combined_data()

    validate_dataset(data)

    modeling_data = prepare_modeling_dataset(data)

    train_data, validation_data, test_data = (
        chronological_split(modeling_data)
    )

    save_modeling_data(
        train_data,
        validation_data,
        test_data,
    )

    print("\nMODELING DATASET")
    print(f"Total rows: {len(modeling_data)}")
    print(
        f"Features: {len(modeling_data.attrs['feature_columns'])}"
    )

    print("\nTRAINING DATA")
    print(train_data.shape)

    print("\nVALIDATION DATA")
    print(validation_data.shape)

    print("\nTEST DATA")
    print(test_data.shape)

    logging.info(
        "Modeling dataset preparation completed."
    )


if __name__ == "__main__":
    main()
