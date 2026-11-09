
"""Merge daily financial news sentiment with market-macro features."""

from pathlib import Path
import logging

import pandas as pd

from src.config import PROJECT_ROOT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MARKET_MACRO_PATH = PROCESSED_DIR / "combined_features.parquet"
SENTIMENT_PATH = PROCESSED_DIR / "news_sentiment_daily.parquet"

OUTPUT_PATH = (
    PROCESSED_DIR / "combined_features_with_sentiment.parquet"
)


def main():
    logging.info("Starting sentiment integration.")

    # --------------------------------------------------
    # 1. Load existing datasets
    # --------------------------------------------------

    market_macro = pd.read_parquet(MARKET_MACRO_PATH)
    sentiment = pd.read_parquet(SENTIMENT_PATH)

    logging.info(
        "Market-macro dataset: %s",
        market_macro.shape,
    )

    logging.info(
        "Daily sentiment dataset: %s",
        sentiment.shape,
    )

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------

    market_required = {"Date", "Ticker"}
    sentiment_required = {
        "Date",
        "Ticker",
        "News_Count",
        "Average_Sentiment",
        "Positive_Count",
        "Neutral_Count",
        "Negative_Count",
        "Average_Confidence",
    }

    missing_market = market_required - set(market_macro.columns)
    missing_sentiment = sentiment_required - set(sentiment.columns)

    if missing_market:
        raise ValueError(
            f"Market dataset missing columns: {missing_market}"
        )

    if missing_sentiment:
        raise ValueError(
            f"Sentiment dataset missing columns: {missing_sentiment}"
        )

    # --------------------------------------------------
    # 3. Normalize merge keys
    # --------------------------------------------------

    market_macro = market_macro.copy()
    sentiment = sentiment.copy()

    market_macro["Date"] = pd.to_datetime(
        market_macro["Date"],
        errors="coerce",
    ).dt.normalize()

    sentiment["Date"] = pd.to_datetime(
        sentiment["Date"],
        errors="coerce",
    ).dt.normalize()

    market_macro["Ticker"] = (
        market_macro["Ticker"].astype(str).str.strip()
    )

    sentiment["Ticker"] = (
        sentiment["Ticker"].astype(str).str.strip()
    )

    if market_macro[["Date", "Ticker"]].isna().any().any():
        raise ValueError(
            "Market dataset contains invalid merge keys."
        )

    if sentiment[["Date", "Ticker"]].isna().any().any():
        raise ValueError(
            "Sentiment dataset contains invalid merge keys."
        )

    # --------------------------------------------------
    # 4. Validate unique keys
    # --------------------------------------------------

    if market_macro.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker keys in market-macro data."
        )

    if sentiment.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker keys in sentiment data."
        )

    original_rows = len(market_macro)

    # --------------------------------------------------
    # 5. Select sentiment columns
    # --------------------------------------------------

    sentiment_columns = [
        "Date",
        "Ticker",
        "News_Count",
        "Average_Sentiment",
        "Positive_Count",
        "Neutral_Count",
        "Negative_Count",
        "Average_Confidence",
    ]

    sentiment = sentiment[sentiment_columns]

    # --------------------------------------------------
    # 6. Merge without losing market-macro observations
    # --------------------------------------------------

    combined = market_macro.merge(
        sentiment,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    combined["Has_News"] = (
        combined["_merge"] == "both"
    ).astype("int8")

    combined.drop(
        columns="_merge",
        inplace=True,
    )

    # --------------------------------------------------
    # 7. Handle dates without news
    # --------------------------------------------------

    count_columns = [
        "News_Count",
        "Positive_Count",
        "Neutral_Count",
        "Negative_Count",
    ]

    sentiment_value_columns = [
        "Average_Sentiment",
        "Average_Confidence",
    ]

    # No news is not the same as neutral news.
    # Has_News preserves this distinction.

    combined[count_columns] = (
        combined[count_columns].fillna(0)
    )

    combined[sentiment_value_columns] = (
        combined[sentiment_value_columns].fillna(0.0)
    )

    # --------------------------------------------------
    # 8. Validate output
    # --------------------------------------------------

    if len(combined) != original_rows:
        raise ValueError(
            "Row count changed during sentiment integration: "
            f"{original_rows} -> {len(combined)}"
        )

    if combined.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker keys after integration."
        )

    if combined.isna().any().any():
        logging.warning(
            "Combined dataset contains missing values. "
            "These may originate from existing market-macro features."
        )

    matched_rows = int(combined["Has_News"].sum())

    logging.info(
        "Market-macro rows: %d",
        original_rows,
    )

    logging.info(
        "Rows matched with sentiment: %d",
        matched_rows,
    )

    logging.info(
        "Rows without sentiment: %d",
        original_rows - matched_rows,
    )

    # --------------------------------------------------
    # 9. Save separate dataset
    # --------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    logging.info(
        "Sentiment-enriched dataset saved: %s",
        OUTPUT_PATH,
    )

    logging.info(
        "Final dataset shape: %s",
        combined.shape,
    )

    print("\n" + "=" * 60)
    print("SENTIMENT INTEGRATION COMPLETED")
    print("=" * 60)
    print(f"Original rows: {original_rows:,}")
    print(f"Final rows: {len(combined):,}")
    print(f"Matched sentiment rows: {matched_rows:,}")
    print(f"Rows without news: {original_rows - matched_rows:,}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
