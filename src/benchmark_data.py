
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "benchmark.parquet"
)

START_DATE = "2015-01-01"
END_DATE = "2026-11-01"

BENCHMARK = "^GSPC"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def download_benchmark():

    logger.info(
        "Downloading S&P 500 benchmark data."
    )

    data = yf.download(
        BENCHMARK,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "Benchmark download returned no data."
        )

    # yfinance may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = [
            column[0]
            for column in data.columns
        ]

    if "Close" not in data.columns:
        raise ValueError(
            f"Close column not found: {list(data.columns)}"
        )

    benchmark = data[
        ["Close"]
    ].copy()

    benchmark = benchmark.reset_index()

    benchmark.columns = [
        "Date",
        "Price",
    ]

    benchmark["Date"] = pd.to_datetime(
        benchmark["Date"]
    )

    benchmark["Ticker"] = BENCHMARK

    benchmark = benchmark[
        [
            "Date",
            "Ticker",
            "Price",
        ]
    ]

    benchmark = benchmark.dropna()

    benchmark = benchmark.sort_values(
        "Date"
    )

    if benchmark.empty:
        raise ValueError(
            "No valid benchmark observations."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Benchmark rows: %s",
        len(benchmark),
    )

    logger.info(
        "Benchmark dates: %s to %s",
        benchmark["Date"].min(),
        benchmark["Date"].max(),
    )

    logger.info(
        "Saved benchmark data: %s",
        OUTPUT_PATH,
    )

    print("\nS&P 500 BENCHMARK DOWNLOAD COMPLETED")
    print(f"Rows: {len(benchmark)}")
    print(f"File: {OUTPUT_PATH}")


if __name__ == "__main__":
    download_benchmark()
