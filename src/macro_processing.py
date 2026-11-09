
"""Macroeconomic data processing and market-calendar alignment."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config, project_path


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_raw_macro_data() -> pd.DataFrame:
    """Load raw FRED observations."""

    config = load_config()

    raw_path = (
        project_path(config["paths"]["raw_data"])
        / "macro.parquet"
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw macro data not found: {raw_path}"
        )

    logging.info("Loading raw macro data: %s", raw_path)

    data = pd.read_parquet(raw_path)

    if data.empty:
        raise ValueError("Raw macro dataset is empty.")

    return data


def validate_macro_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the required macroeconomic columns."""

    config = load_config()

    required_columns = (
        ["Date"] + config["data"]["fred_series"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing macro columns: {missing_columns}"
        )

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    if data["Date"].isna().any():
        raise ValueError(
            "Macro dataset contains invalid dates."
        )

    data = data.sort_values("Date").reset_index(drop=True)

    if data["Date"].duplicated().any():
        raise ValueError(
            "Macro dataset contains duplicate dates."
        )

    logging.info(
        "Macro validation successful: %d rows.",
        len(data),
    )

    return data


def prepare_treasury_yields(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare daily Treasury yields."""

    yields = data[
        ["Date", "DGS10", "DGS3MO"]
    ].copy()

    yields = yields.sort_values("Date")

    # Carry the latest published yield forward.
    # No backward filling is performed.
    yields[["DGS10", "DGS3MO"]] = (
        yields[["DGS10", "DGS3MO"]]
        .ffill()
    )

    # Treasury yield spread.
    yields["Treasury_Spread_10Y_3M"] = (
        yields["DGS10"] - yields["DGS3MO"]
    )

    return yields


def prepare_monthly_macro(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare monthly CPI and unemployment features.

    Availability date is approximated as the observation date
    plus two calendar months. This is not an exact release-date
    reconstruction.
    """

    monthly = data[
        ["Date", "CPIAUCSL", "UNRATE"]
    ].copy()

    monthly = monthly.dropna(
        subset=["CPIAUCSL", "UNRATE"],
        how="all",
    )

    monthly = monthly.sort_values("Date")

    # Conservative approximate availability date.
    monthly["Date"] = (
        monthly["Date"] + pd.DateOffset(months=2)
    )

    monthly = monthly.sort_values("Date")

    if monthly["Date"].duplicated().any():
        raise ValueError(
            "Duplicate monthly macro availability dates."
        )

    # CPI year-over-year percentage change.
    monthly["CPI_YoY"] = (
        monthly["CPIAUCSL"].pct_change(periods=12) * 100
    )

    # Monthly unemployment rate change.
    monthly["UNRATE_Change"] = (
        monthly["UNRATE"].diff()
    )

    monthly = monthly[
        [
            "Date",
            "CPIAUCSL",
            "CPI_YoY",
            "UNRATE",
            "UNRATE_Change",
        ]
    ]

    return monthly


def align_macro_to_market_calendar(
    treasury: pd.DataFrame,
    monthly: pd.DataFrame,
    market_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Align macro observations to market dates."""

    calendar = (
        market_dates[["Date"]]
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    treasury = treasury.sort_values("Date")

    monthly = monthly.sort_values("Date")

    # Align daily Treasury observations.
    aligned = pd.merge_asof(
        calendar,
        treasury,
        on="Date",
        direction="backward",
    )

    # Align monthly series using their approximate
    # availability dates.
    aligned = pd.merge_asof(
        aligned.sort_values("Date"),
        monthly,
        on="Date",
        direction="backward",
    )

    aligned = aligned.sort_values("Date").reset_index(drop=True)

    return aligned


def process_macro_data() -> pd.DataFrame:
    """Run macro preprocessing and save the aligned dataset."""

    config = load_config()

    logging.info("Starting macroeconomic processing.")

    macro = load_raw_macro_data()

    macro = validate_macro_data(macro)

    treasury = prepare_treasury_yields(macro)

    monthly = prepare_monthly_macro(macro)

    market_path = (
        project_path(config["paths"]["raw_data"])
        / "prices.parquet"
    )

    if not market_path.exists():
        raise FileNotFoundError(
            f"Market data not found: {market_path}"
        )

    market = pd.read_parquet(market_path)

    market_dates = market[["Date"]].copy()

    market_dates["Date"] = pd.to_datetime(
        market_dates["Date"]
    )

    processed = align_macro_to_market_calendar(
        treasury=treasury,
        monthly=monthly,
        market_dates=market_dates,
    )

    output_dir = project_path(
        config["paths"]["processed_data"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "macro_features.parquet"

    processed.to_parquet(
        output_path,
        index=False,
    )

    logging.info(
        "Processed macro dataset saved: %s",
        output_path,
    )

    logging.info(
        "Processed macro dataset shape: %s",
        processed.shape,
    )

    logging.info(
        "Macro processing completed successfully."
    )

    return processed


def main() -> None:
    configure_logging()

    process_macro_data()


if __name__ == "__main__":
    main()
