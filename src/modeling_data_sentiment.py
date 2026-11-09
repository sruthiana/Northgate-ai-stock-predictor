
"""Prepare chronological modeling datasets with news sentiment."""

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
    """Load market, macroeconomic, and sentiment features."""

    config = load_config()

    data_path = (
        project_path(config["paths"]["processed_data"])
        / "combined_features_with_sentiment.parquet"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Sentiment dataset not found: {data_path}"
        )

    data = pd.read_parquet(data_path)

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    logging.info("Loaded sentiment dataset: %s", data.shape)

    return data


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate dataset before modeling."""

    required_columns = [
        "Date",
        "Ticker",
        "Target_Next_Day_Log_Return",
        "Has_News",
        "News_Count",
        "Average_Sentiment",
        "Average_Confidence",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
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

    logging.info("Sentiment dataset validation passed.")


def prepare_modeling_dataset(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare numeric features including sentiment."""

    config = load_config()

    equities = config["data"]["equity_universe"]

    data = data[
        data["Ticker"].isin(equities)
    ].copy()

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    data = data.dropna(
        subset=["Target_Next_Day_Log_Return"]
    )

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

    numeric_features = data[
        feature_columns
    ].select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_features:
        raise ValueError(
            "No numeric features available."
        )

    sentiment_features = [
        "Has_News",
        "News_Count",
        "Average_Sentiment",
        "Positive_Count",
        "Neutral_Count",
        "Negative_Count",
        "Average_Confidence",
    ]

    missing_sentiment = [
        column
        for column in sentiment_features
        if column not in numeric_features
    ]

    if missing_sentiment:
        raise ValueError(
            f"Sentiment features missing: {missing_sentiment}"
        )

    data[numeric_features] = data[
        numeric_features
    ].replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    logging.info(
        "Numeric features selected: %s",
        len(numeric_features),
    )

    logging.info(
        "Rows before missing-value filtering: %s",
        len(data),
    )

    data = data.dropna(
        subset=numeric_features
    ).copy()

    logging.info(
        "Rows after missing-value filtering: %s",
        len(data),
    )

    data.attrs["feature_columns"] = numeric_features

    return data


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train, validation, and test datasets."""

    config = load_config()

    test_fraction = config["modeling"]["test_fraction"]

    unique_dates = sorted(
        data["Date"].drop_duplicates()
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "Not enough unique dates for splitting."
        )

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
        "Train: %s to %s",
        train_data["Date"].min(),
        train_data["Date"].max(),
    )

    logging.info(
        "Validation: %s to %s",
        validation_data["Date"].min(),
        validation_data["Date"].max(),
    )

    logging.info(
        "Test: %s to %s",
        test_data["Date"].min(),
        test_data["Date"].max(),
    )

    return train_data, validation_data, test_data


def save_modeling_data(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save sentiment datasets without overwriting baseline files."""

    config = load_config()

    output_dir = project_path(
        config["paths"]["processed_data"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_parquet(
        output_dir / "train_sentiment.parquet",
        index=False,
    )

    validation_data.to_parquet(
        output_dir / "validation_sentiment.parquet",
        index=False,
    )

    test_data.to_parquet(
        output_dir / "test_sentiment.parquet",
        index=False,
    )

    logging.info(
        "Sentiment train, validation, and test files saved."
    )


def main() -> None:
    configure_logging()

    logging.info(
        "Starting sentiment modeling dataset preparation."
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

    print("\nSENTIMENT MODELING DATASET")
    print(f"Total rows: {len(modeling_data)}")
    print(
        f"Features: "
        f"{len(modeling_data.attrs['feature_columns'])}"
    )

    print("\nTRAIN:", train_data.shape)
    print("VALIDATION:", validation_data.shape)
    print("TEST:", test_data.shape)

    print("\nSentiment feature columns:")
    print(
        modeling_data.attrs["feature_columns"]
    )

    logging.info(
        "Sentiment modeling dataset preparation completed."
    )


if __name__ == "__main__":
    main()
