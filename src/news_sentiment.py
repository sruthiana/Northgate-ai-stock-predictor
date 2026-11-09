
"""Financial news collection and FinBERT sentiment analysis."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from transformers import pipeline

from src.config import load_config, project_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

MODEL_NAME = "ProsusAI/finbert"

COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson Johnson",
    "PG": "Procter Gamble",
    "NVDA": "NVIDIA",
    "KO": "Coca Cola",
    "CAT": "Caterpillar",
    "HD": "Home Depot",
}

RSS_URL = "https://news.google.com/rss/search"


def get_news_url(query: str) -> str:
    """Build Google News RSS search URL."""

    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }

    return f"{RSS_URL}?{urlencode(params)}"


def parse_publication_date(value: str) -> datetime:
    """Parse RSS publication date into UTC."""

    try:
        parsed = parsedate_to_datetime(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except Exception:
        return datetime.now(timezone.utc)


def fetch_news_for_ticker(
    ticker: str,
    company: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Fetch RSS headlines for one company."""

    query = f'"{company}" stock'

    url = get_news_url(query)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "NorthgateStockPredictor/1.0"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    articles = []

    for item in root.findall(".//item"):

        title_element = item.find("title")
        link_element = item.find("link")
        date_element = item.find("pubDate")
        source_element = item.find("source")

        if (
            title_element is None
            or link_element is None
            or date_element is None
        ):
            continue

        title = (title_element.text or "").strip()

        link = (link_element.text or "").strip()

        published = parse_publication_date(
            date_element.text or ""
        )

        published_timestamp = pd.Timestamp(published)

        if not title or not link:
            continue

        if not start <= published_timestamp < end:
            continue

        articles.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Headline": title,
                "URL": link,
                "Published_UTC": published,
                "Source": (
                    source_element.text
                    if source_element is not None
                    else None
                ),
            }
        )

    return articles


def collect_news(config: dict) -> pd.DataFrame:
    """Collect financial headlines for configured equities."""

    data_config = config["data"]

    tickers = data_config["equity_universe"]

    start_date = data_config["start_date"]

    end_date = data_config["end_date"]

    collected = []

    logging.info(
        "Starting financial news collection."
    )

    for ticker in tickers:

        company = COMPANIES.get(ticker, ticker)

        logging.info(
            "Collecting news for %s (%s).",
            company,
            ticker,
        )

        try:

            articles = fetch_news_for_ticker(
                ticker=ticker,
                company=company,
                start_date=start_date,
                end_date=end_date,
            )

            collected.extend(articles)

            logging.info(
                "%s: collected %d headlines.",
                ticker,
                len(articles),
            )

        except Exception as exc:

            logging.warning(
                "News collection failed for %s: %s",
                ticker,
                exc,
            )

        time.sleep(1)

    if not collected:

        raise RuntimeError(
            "No news articles were collected. "
            "Check internet access and RSS availability."
        )

    news = pd.DataFrame(collected)

    news["Published_UTC"] = pd.to_datetime(
        news["Published_UTC"],
        utc=True,
    )

    news = news.drop_duplicates(
        subset=["Ticker", "Headline", "Published_UTC"]
    )

    news = news.sort_values(
        ["Ticker", "Published_UTC"]
    ).reset_index(drop=True)

    logging.info(
        "News collection completed: %d unique articles.",
        len(news),
    )

    return news


def load_finbert():
    """Load pretrained financial sentiment classifier."""

    logging.info(
        "Loading FinBERT model: %s",
        MODEL_NAME,
    )

    # classifier = pipeline(
    #     task="text-classification",
    #     model=MODEL_NAME,
    #     tokenizer=MODEL_NAME,
    #     device=-1,
    # )
    
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        framework="pt",
        device=-1,
    )

    logging.info(
        "FinBERT loaded successfully."
    )

    return classifier



def analyze_sentiment(
    news: pd.DataFrame,
    classifier,
) -> pd.DataFrame:
    """Classify news headlines with FinBERT."""

    news = news.copy()

    headlines = news["Headline"].fillna("").astype(str).tolist()

    logging.info(
        "Analyzing sentiment for %d headlines.",
        len(headlines),
    )

    predictions = classifier(
        headlines,
        batch_size=16,
        truncation=True,
        max_length=256,
    )

    labels = []
    scores = []

    for prediction in predictions:

        label = prediction["label"].lower()

        confidence = float(
            prediction["score"]
        )

        labels.append(label)

        scores.append(confidence)

    # --------------------------------------------------
    # 1. Store FinBERT predictions
    # --------------------------------------------------

    news["Sentiment_Label"] = labels

    news["Sentiment_Confidence"] = pd.to_numeric(
        scores,
        errors="coerce",
    )

    # --------------------------------------------------
    # 2. Calculate directional sentiment
    # --------------------------------------------------

    news["Sentiment_Score"] = 0.0

    positive_mask = (
        news["Sentiment_Label"] == "positive"
    )

    negative_mask = (
        news["Sentiment_Label"] == "negative"
    )

    news.loc[
        positive_mask,
        "Sentiment_Score",
    ] = news.loc[
        positive_mask,
        "Sentiment_Confidence",
    ]

    news.loc[
        negative_mask,
        "Sentiment_Score",
    ] = -news.loc[
        negative_mask,
        "Sentiment_Confidence",
    ]

    # --------------------------------------------------
    # 3. Validate sentiment output
    # --------------------------------------------------

    valid_labels = {
        "positive",
        "neutral",
        "negative",
    }

    invalid_labels = (
        set(news["Sentiment_Label"].unique())
        - valid_labels
    )

    if invalid_labels:

        raise ValueError(
            f"Unexpected FinBERT labels: {invalid_labels}"
        )

    if news["Sentiment_Score"].isna().any():

        raise ValueError(
            "Sentiment scores contain missing values."
        )

    logging.info(
        "Sentiment analysis completed."
    )

    logging.info(
        "Sentiment distribution:\n%s",
        news["Sentiment_Label"].value_counts(),
    )

    return news

def aggregate_daily_sentiment(
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate headline sentiment by ticker and UTC date."""

    news = news.copy()

    # --------------------------------------------------
    # 1. Validate required columns
    # --------------------------------------------------

    required_columns = [
        "Published_UTC",
        "Ticker",
        "Headline",
        "Sentiment_Label",
        "Sentiment_Score",
        "Sentiment_Confidence",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in news.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing sentiment columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 2. Normalize date and numeric columns
    # --------------------------------------------------

    news["Published_UTC"] = pd.to_datetime(
        news["Published_UTC"],
        utc=True,
        errors="coerce",
    )

    news["Sentiment_Score"] = pd.to_numeric(
        news["Sentiment_Score"],
        errors="coerce",
    )

    news["Sentiment_Confidence"] = pd.to_numeric(
        news["Sentiment_Confidence"],
        errors="coerce",
    )

    news = news.dropna(
        subset=[
            "Published_UTC",
            "Ticker",
            "Sentiment_Score",
            "Sentiment_Confidence",
        ]
    )

    if news.empty:
        raise ValueError(
            "No valid sentiment records available for aggregation."
        )

    # --------------------------------------------------
    # 3. Convert timestamps to UTC dates
    # --------------------------------------------------

    news["Date"] = (
        news["Published_UTC"]
        .dt.date
    )

    # --------------------------------------------------
    # 4. Create numeric count columns
    # --------------------------------------------------

    news["Positive_Count"] = (
        news["Sentiment_Label"] == "positive"
    ).astype("int64")

    news["Neutral_Count"] = (
        news["Sentiment_Label"] == "neutral"
    ).astype("int64")

    news["Negative_Count"] = (
        news["Sentiment_Label"] == "negative"
    ).astype("int64")

    # --------------------------------------------------
    # 5. Aggregate by Date and Ticker
    # --------------------------------------------------

    daily = (
        news.groupby(
            ["Date", "Ticker"],
            as_index=False,
        )
        .agg(
            News_Count=(
                "Headline",
                "count",
            ),
            Average_Sentiment=(
                "Sentiment_Score",
                "mean",
            ),
            Positive_Count=(
                "Positive_Count",
                "sum",
            ),
            Neutral_Count=(
                "Neutral_Count",
                "sum",
            ),
            Negative_Count=(
                "Negative_Count",
                "sum",
            ),
            Average_Confidence=(
                "Sentiment_Confidence",
                "mean",
            ),
        )
    )

    # --------------------------------------------------
    # 6. Clean output types
    # --------------------------------------------------

    daily["Date"] = pd.to_datetime(
        daily["Date"],
    )

    numeric_columns = [
        "News_Count",
        "Average_Sentiment",
        "Positive_Count",
        "Neutral_Count",
        "Negative_Count",
        "Average_Confidence",
    ]

    daily[numeric_columns] = daily[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    daily["News_Count"] = daily[
        "News_Count"
    ].astype("int64")

    daily["Positive_Count"] = daily[
        "Positive_Count"
    ].astype("int64")

    daily["Neutral_Count"] = daily[
        "Neutral_Count"
    ].astype("int64")

    daily["Negative_Count"] = daily[
        "Negative_Count"
    ].astype("int64")

    # --------------------------------------------------
    # 7. Sort and validate
    # --------------------------------------------------

    daily = (
        daily.sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    if daily.empty:
        raise ValueError(
            "Daily sentiment aggregation produced no records."
        )

    logging.info(
        "Daily sentiment dataset created: %s",
        daily.shape,
    )

    logging.info(
        "Daily sentiment columns: %s",
        daily.columns.tolist(),
    )

    return daily

def save_outputs(
    news: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict,
) -> None:
    """Save headline-level and daily sentiment datasets."""

    raw_dir = project_path(
        config["paths"]["raw_data"]
    )

    processed_dir = project_path(
        config["paths"]["processed_data"]
    )

    reports_dir = project_path(
        config["paths"]["reports"]
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = raw_dir / "news_articles.csv"

    daily_path = (
        processed_dir / "news_sentiment_daily.parquet"
    )

    summary_path = (
        reports_dir / "news_sentiment_summary.csv"
    )

    news.to_csv(
        raw_path,
        index=False,
    )

    daily.to_parquet(
        daily_path,
        index=False,
    )

    summary = daily.groupby(
        "Ticker",
        as_index=False,
    ).agg(
        Total_News=("News_Count", "sum"),
        Average_Sentiment=("Average_Sentiment", "mean"),
        Positive_Count=("Positive_Count", "sum"),
        Neutral_Count=("Neutral_Count", "sum"),
        Negative_Count=("Negative_Count", "sum"),
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    logging.info(
        "Saved headline dataset: %s",
        raw_path,
    )

    logging.info(
        "Saved daily sentiment dataset: %s",
        daily_path,
    )

    logging.info(
        "Saved sentiment summary: %s",
        summary_path,
    )


def main() -> None:
    """Execute the complete news sentiment pipeline."""

    config = load_config()

    logging.info(
        "Starting Northgate financial news sentiment pipeline."
    )

    news = collect_news(config)

    classifier = load_finbert()

    analyzed_news = analyze_sentiment(
        news,
        classifier,
    )

    daily = aggregate_daily_sentiment(
        analyzed_news,
    )

    save_outputs(
        analyzed_news,
        daily,
        config,
    )

    logging.info(
        "News sentiment pipeline completed successfully."
    )

    print("\n" + "=" * 70)

    print("NEWS SENTIMENT PIPELINE COMPLETED")

    print("=" * 70)

    print(
        f"Articles analyzed: {len(analyzed_news):,}"
    )

    print(
        f"Daily ticker observations: {len(daily):,}"
    )

    print(
        f"Tickers covered: {daily['Ticker'].nunique()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
