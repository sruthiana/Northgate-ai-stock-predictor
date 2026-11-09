
"""Market data preprocessing and technical feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import load_config, project_path


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_raw_market_data() -> pd.DataFrame:
    """Load raw market data from Parquet."""

    config = load_config()

    raw_path = project_path(
        config["paths"]["raw_data"]
    ) / "prices.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw market data not found: {raw_path}"
        )

    logging.info("Loading raw market data: %s", raw_path)

    data = pd.read_parquet(raw_path)

    if data.empty:
        raise ValueError("Raw market dataset is empty.")

    return data


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and market records."""

    required_columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
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

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    if data["Date"].isna().any():
        raise ValueError("Date column contains missing values.")

    if data["Ticker"].isna().any():
        raise ValueError("Ticker column contains missing values.")

    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    duplicates = data.duplicated(
        subset=["Ticker", "Date"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate ticker-date records."
        )

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]

    if data[price_columns].isna().any().any():
        raise ValueError("Market prices contain missing values.")

    if (data[price_columns] <= 0).any().any():
        raise ValueError("Market prices must be positive.")

    if (data["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    invalid_ohlc = (
        (data["High"] < data["Low"])
        | (data["High"] < data["Open"])
        | (data["High"] < data["Close"])
        | (data["Low"] > data["Open"])
        | (data["Low"] > data["Close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            f"Found {invalid_ohlc.sum()} invalid OHLC records."
        )

    logging.info(
        "Validation successful: %d rows, %d tickers.",
        len(data),
        data["Ticker"].nunique(),
    )

    return data


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple and logarithmic returns."""

    data = data.copy()

    data["Daily_Return"] = (
        data.groupby("Ticker")["Adj Close"]
        .pct_change()
    )

    data["Log_Return"] = (
        np.log(data["Adj Close"])
        .groupby(data["Ticker"])
        .diff()
    )

    return data


def calculate_moving_averages(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate moving averages using adjusted close."""

    data = data.copy()

    grouped = data.groupby("Ticker")["Adj Close"]

    data["SMA_20"] = grouped.transform(
        lambda series: series.rolling(
            window=20,
            min_periods=20,
        ).mean()
    )

    data["SMA_50"] = grouped.transform(
        lambda series: series.rolling(
            window=50,
            min_periods=50,
        ).mean()
    )

    data["SMA_200"] = grouped.transform(
        lambda series: series.rolling(
            window=200,
            min_periods=200,
        ).mean()
    )

    data["EMA_12"] = grouped.transform(
        lambda series: series.ewm(
            span=12,
            adjust=False,
            min_periods=12,
        ).mean()
    )

    data["EMA_26"] = grouped.transform(
        lambda series: series.ewm(
            span=26,
            adjust=False,
            min_periods=26,
        ).mean()
    )

    return data


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""

    delta = prices.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain / average_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    # Handle periods with gains but no losses.
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )

    # Handle periods with no gains and no losses.
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50.0,
    )

    return rsi


def calculate_technical_indicators(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate RSI, MACD, Bollinger Bands, and volatility."""

    data = data.copy()

    grouped = data.groupby("Ticker")["Adj Close"]

    data["RSI_14"] = grouped.transform(
        lambda series: calculate_rsi(series, period=14)
    )

    # MACD
    data["MACD"] = (
        data["EMA_12"] - data["EMA_26"]
    )

    data["MACD_Signal"] = (
        data.groupby("Ticker")["MACD"]
        .transform(
            lambda series: series.ewm(
                span=9,
                adjust=False,
                min_periods=9,
            ).mean()
        )
    )

    data["MACD_Histogram"] = (
        data["MACD"] - data["MACD_Signal"]
    )

    # Bollinger Bands
    data["BB_Middle"] = grouped.transform(
        lambda series: series.rolling(
            window=20,
            min_periods=20,
        ).mean()
    )

    data["BB_Std"] = grouped.transform(
        lambda series: series.rolling(
            window=20,
            min_periods=20,
        ).std()
    )

    data["BB_Upper"] = (
        data["BB_Middle"] + 2 * data["BB_Std"]
    )

    data["BB_Lower"] = (
        data["BB_Middle"] - 2 * data["BB_Std"]
    )

    # Rolling volatility based on daily returns
    data["Volatility_20"] = (
        data.groupby("Ticker")["Daily_Return"]
        .transform(
            lambda series: series.rolling(
                window=20,
                min_periods=20,
            ).std()
        )
    )

    data["Volatility_60"] = (
        data.groupby("Ticker")["Daily_Return"]
        .transform(
            lambda series: series.rolling(
                window=60,
                min_periods=60,
            ).std()
        )
    )

    # Volume moving average
    data["Volume_SMA_20"] = (
        data.groupby("Ticker")["Volume"]
        .transform(
            lambda series: series.rolling(
                window=20,
                min_periods=20,
            ).mean()
        )
    )

    return data


def calculate_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create next-day log-return prediction target."""

    data = data.copy()

    data["Target_Next_Day_Log_Return"] = (
        data.groupby("Ticker")["Log_Return"]
        .shift(-1)
    )

    return data


def process_market_data() -> pd.DataFrame:
    """Run the complete market-data processing pipeline."""

    config = load_config()

    logging.info("Starting market-data processing.")

    data = load_raw_market_data()

    data = validate_market_data(data)

    data = calculate_returns(data)

    data = calculate_moving_averages(data)

    data = calculate_technical_indicators(data)

    data = calculate_target(data)

    processed_dir = project_path(
        config["paths"]["processed_data"]
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        processed_dir / "market_features.parquet"
    )

    data.to_parquet(
        output_path,
        index=False,
    )

    logging.info(
        "Processed dataset saved: %s",
        output_path,
    )

    logging.info(
        "Processed dataset shape: %s",
        data.shape,
    )

    logging.info(
        "Processing completed successfully."
    )

    return data


def main() -> None:
    configure_logging()

    process_market_data()


if __name__ == "__main__":
    main()
