
"""Download and validate raw market and macroeconomic data."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

from src.config import load_config, PROJECT_ROOT, project_path


def configure_logging(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                report_dir / "data_ingestion_log.txt",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def configure_yfinance_cache() -> None:
    """Use a dedicated directory for Yahoo Finance timezone caching."""

    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        yf.set_tz_cache_location(str(cache_dir))
        logging.info("Yahoo Finance cache configured: %s", cache_dir)
    except Exception as exc:
        logging.warning("Could not configure Yahoo Finance cache: %s", exc)


def download_one_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    attempts: int = 3,
) -> pd.DataFrame:
    """Download one symbol with retry handling."""

    for attempt in range(1, attempts + 1):
        try:
            logging.info(
                "Downloading %s (attempt %d/%d)",
                ticker,
                attempt,
                attempts,
            )

            downloaded = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                timeout=30,
            )

            if downloaded is None or downloaded.empty:
                raise RuntimeError("Yahoo Finance returned no rows.")

            if isinstance(downloaded.columns, pd.MultiIndex):
                downloaded.columns = downloaded.columns.get_level_values(0)

            downloaded.columns = [
                str(column) for column in downloaded.columns
            ]

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]

            missing = [
                column for column in required
                if column not in downloaded.columns
            ]

            if missing:
                raise RuntimeError(
                    f"Missing columns: {missing}"
                )

            result = downloaded[required].copy()

            result.index.name = "Date"
            result = result.reset_index()

            result["Date"] = pd.to_datetime(
                result["Date"]
            ).dt.tz_localize(None)

            result["Ticker"] = ticker

            result = result[
                [
                    "Date",
                    "Ticker",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                ]
            ]

            result = result.dropna(
                subset=["Date", "Adj Close"]
            )

            if result.empty:
                raise RuntimeError(
                    "No valid observations after cleaning."
                )

            logging.info(
                "%s downloaded successfully: %d rows",
                ticker,
                len(result),
            )

            return result

        except Exception as exc:
            logging.warning(
                "%s download failed on attempt %d: %s",
                ticker,
                attempt,
                exc,
            )

            if attempt < attempts:
                time.sleep(2 * attempt)

    logging.error(
        "All download attempts failed for %s",
        ticker,
    )

    return pd.DataFrame()


def download_market_data(config: dict) -> pd.DataFrame:
    """Download all configured equities, benchmark, and VIX."""

    data_cfg = config["data"]

    equities = data_cfg["equity_universe"]

    symbols = (
        equities
        + [data_cfg["benchmark"]]
        + [data_cfg["volatility_index"]]
    )

    configure_yfinance_cache()

    logging.info(
        "Downloading %d symbols individually.",
        len(symbols),
    )

    frames = []
    failed_symbols = []

    for ticker in symbols:

        result = download_one_ticker(
            ticker=ticker,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
        )

        if result.empty:
            failed_symbols.append(ticker)
        else:
            frames.append(result)

    if not frames:
        raise RuntimeError(
            "No market data could be downloaded."
        )

    market = pd.concat(
        frames,
        ignore_index=True,
    )

    market = market.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    actual_symbols = set(market["Ticker"].unique())

    missing_symbols = [
        ticker for ticker in symbols
        if ticker not in actual_symbols
    ]

    logging.info(
        "Market data received: %s rows, %d symbols.",
        f"{len(market):,}",
        market["Ticker"].nunique(),
    )

    if missing_symbols:
        logging.error(
            "Missing required symbols: %s",
            ", ".join(missing_symbols),
        )

        raise RuntimeError(
            "Market ingestion incomplete. Missing symbols: "
            + ", ".join(missing_symbols)
        )

    logging.info(
        "All %d required symbols downloaded successfully.",
        len(symbols),
    )

    return market


def download_fred_data(config: dict) -> pd.DataFrame:
    """Download raw FRED observations."""

    data_cfg = config["data"]

    series = data_cfg["fred_series"]

    logging.info(
        "Downloading FRED series: %s",
        ", ".join(series),
    )

    macro = web.DataReader(
        series,
        "fred",
        data_cfg["start_date"],
        data_cfg["end_date"],
    )

    if macro is None or macro.empty:
        raise RuntimeError(
            "FRED returned no macroeconomic observations."
        )

    macro.index.name = "Date"

    macro = macro.reset_index()

    macro["Date"] = pd.to_datetime(
        macro["Date"]
    ).dt.tz_localize(None)

    logging.info(
        "FRED data received: %s rows, %d series.",
        f"{len(macro):,}",
        len(series),
    )

    return macro


def main() -> None:
    config = load_config()

    raw_dir = project_path(
        config["paths"]["raw_data"]
    )

    report_dir = project_path(
        config["paths"]["reports"]
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_logging(report_dir)

    logging.info(
        "Starting Northgate raw-data ingestion."
    )

    market = download_market_data(config)

    macro = download_fred_data(config)

    market_path = raw_dir / "prices.parquet"

    macro_path = raw_dir / "macro.parquet"

    market.to_parquet(
        market_path,
        index=False,
    )

    macro.to_parquet(
        macro_path,
        index=False,
    )

    logging.info(
        "Saved market data: %s",
        market_path.relative_to(PROJECT_ROOT),
    )

    logging.info(
        "Saved macro data: %s",
        macro_path.relative_to(PROJECT_ROOT),
    )

    logging.info(
        "Ingestion completed successfully."
    )


if __name__ == "__main__":
    main()
