
"""Combine processed market and macroeconomic features."""

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


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load market features and macro features."""

    config = load_config()

    processed_dir = project_path(
        config["paths"]["processed_data"]
    )

    market_path = processed_dir / "market_features.parquet"

    macro_path = processed_dir / "macro_features.parquet"

    if not market_path.exists():
        raise FileNotFoundError(
            f"Market features not found: {market_path}"
        )

    if not macro_path.exists():
        raise FileNotFoundError(
            f"Macro features not found: {macro_path}"
        )

    market = pd.read_parquet(market_path)

    macro = pd.read_parquet(macro_path)

    logging.info(
        "Market features loaded: %s",
        market.shape,
    )

    logging.info(
        "Macro features loaded: %s",
        macro.shape,
    )

    return market, macro


def validate_inputs(
    market: pd.DataFrame,
    macro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate dates and uniqueness before merging."""

    market = market.copy()

    macro = macro.copy()

    market["Date"] = pd.to_datetime(
        market["Date"]
    )

    macro["Date"] = pd.to_datetime(
        macro["Date"]
    )

    if market["Date"].isna().any():
        raise ValueError(
            "Market data contains missing dates."
        )

    if macro["Date"].isna().any():
        raise ValueError(
            "Macro data contains missing dates."
        )

    if market.duplicated(
        subset=["Ticker", "Date"]
    ).any():
        raise ValueError(
            "Market data contains duplicate ticker-date records."
        )

    if macro["Date"].duplicated().any():
        raise ValueError(
            "Macro data contains duplicate dates."
        )

    market = market.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    macro = macro.sort_values(
        "Date"
    ).reset_index(drop=True)

    return market, macro


def merge_features(
    market: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Merge macro features onto market observations."""

    logging.info("Starting market-macro merge.")

    market, macro = validate_inputs(
        market,
        macro,
    )

    original_market_rows = len(market)

    original_market_keys = market[
        ["Ticker", "Date"]
    ].copy()

    merged = market.merge(
        macro,
        on="Date",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = (
        merged["_merge"] != "both"
    ).sum()

    if unmatched:
        logging.warning(
            "%d market rows did not match macro dates.",
            unmatched,
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    if len(merged) != original_market_rows:
        raise RuntimeError(
            "Merge changed the market row count."
        )

    if merged.duplicated(
        subset=["Ticker", "Date"]
    ).any():
        raise RuntimeError(
            "Merge introduced duplicate ticker-date records."
        )

    merged_keys = merged[
        ["Ticker", "Date"]
    ].reset_index(drop=True)

    original_market_keys = (
        original_market_keys.reset_index(drop=True)
    )

    if not merged_keys.equals(
        original_market_keys
    ):
        raise RuntimeError(
            "Market ticker-date records changed during merge."
        )

    logging.info(
        "Merge successful: %d rows, %d columns.",
        merged.shape[0],
        merged.shape[1],
    )

    return merged


def main() -> None:
    configure_logging()

    config = load_config()

    market, macro = load_processed_data()

    combined = merge_features(
        market,
        macro,
    )

    output_dir = project_path(
        config["paths"]["processed_data"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "combined_features.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False,
    )

    logging.info(
        "Combined dataset saved: %s",
        output_path,
    )

    logging.info(
        "Combined dataset shape: %s",
        combined.shape,
    )

    logging.info(
        "Feature merge completed successfully."
    )


if __name__ == "__main__":
    main()
