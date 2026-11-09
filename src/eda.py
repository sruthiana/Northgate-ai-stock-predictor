
"""Exploratory data analysis for Northgate AI Stock Predictor."""

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


def load_combined_data() -> pd.DataFrame:
    """Load the combined feature dataset."""

    config = load_config()

    path = (
        project_path(config["paths"]["processed_data"])
        / "combined_features.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Combined dataset not found: {path}"
        )

    data = pd.read_parquet(path)

    data["Date"] = pd.to_datetime(data["Date"])

    logging.info(
        "Loaded combined dataset: %s",
        data.shape,
    )

    return data


def calculate_return_statistics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive return statistics by ticker."""

    data = data.copy()

    data = data.dropna(
        subset=["Daily_Return", "Log_Return"]
    )

    statistics = (
        data.groupby("Ticker")
        .agg(
            observations=("Daily_Return", "count"),
            mean_daily_return=("Daily_Return", "mean"),
            median_daily_return=("Daily_Return", "median"),
            daily_return_std=("Daily_Return", "std"),
            minimum_daily_return=("Daily_Return", "min"),
            maximum_daily_return=("Daily_Return", "max"),
            mean_log_return=("Log_Return", "mean"),
            daily_log_return_std=("Log_Return", "std"),
        )
        .reset_index()
    )

    statistics["annualized_volatility"] = (
        statistics["daily_return_std"] * np.sqrt(252)
    )

    statistics["annualized_mean_daily_return"] = (
        statistics["mean_daily_return"] * 252
    )

    statistics["annualized_mean_log_return"] = (
        statistics["mean_log_return"] * 252
    )

    return statistics


def calculate_equity_performance(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate cumulative performance using adjusted close."""

    config = load_config()

    equities = config["data"]["equity_universe"]

    data = data[
        data["Ticker"].isin(equities)
    ].copy()

    data = data.sort_values(
        ["Ticker", "Date"]
    )

    performance = []

    for ticker, group in data.groupby("Ticker"):

        group = group.sort_values("Date")

        first_price = group["Adj Close"].iloc[0]

        last_price = group["Adj Close"].iloc[-1]

        cumulative_return = (
            last_price / first_price
        ) - 1

        daily_returns = group["Daily_Return"].dropna()

        annualized_volatility = (
            daily_returns.std() * np.sqrt(252)
        )

        cumulative_returns = (
            group["Adj Close"] / first_price
        )

        running_max = cumulative_returns.cummax()

        drawdown = (
            cumulative_returns / running_max
        ) - 1

        maximum_drawdown = drawdown.min()

        performance.append(
            {
                "Ticker": ticker,
                "First_Date": group["Date"].iloc[0],
                "Last_Date": group["Date"].iloc[-1],
                "Starting_Adj_Close": first_price,
                "Ending_Adj_Close": last_price,
                "Cumulative_Return": cumulative_return,
                "Annualized_Volatility": annualized_volatility,
                "Maximum_Drawdown": maximum_drawdown,
            }
        )

    result = pd.DataFrame(performance)

    result = result.sort_values(
        "Cumulative_Return",
        ascending=False,
    ).reset_index(drop=True)

    return result


def calculate_return_correlations(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate correlations between equity daily returns."""

    config = load_config()

    equities = config["data"]["equity_universe"]

    data = data[
        data["Ticker"].isin(equities)
    ].copy()

    return_matrix = data.pivot(
        index="Date",
        columns="Ticker",
        values="Daily_Return",
    )

    correlation = return_matrix.corr()

    return correlation


def save_eda_results(
    statistics: pd.DataFrame,
    performance: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Save EDA tables as CSV files."""

    config = load_config()

    reports_dir = project_path(
        config["paths"]["reports"]
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics.to_csv(
        reports_dir / "return_statistics.csv",
        index=False,
    )

    performance.to_csv(
        reports_dir / "equity_performance.csv",
        index=False,
    )

    correlation.to_csv(
        reports_dir / "equity_return_correlation.csv"
    )

    logging.info(
        "EDA tables saved to %s",
        reports_dir,
    )


def main() -> None:
    configure_logging()

    logging.info("Starting exploratory data analysis.")

    data = load_combined_data()

    statistics = calculate_return_statistics(data)

    performance = calculate_equity_performance(data)

    correlation = calculate_return_correlations(data)

    save_eda_results(
        statistics,
        performance,
        correlation,
    )

    print("\nRETURN STATISTICS")
    print(statistics.to_string(index=False))

    print("\nEQUITY PERFORMANCE")
    print(performance.to_string(index=False))

    print("\nEQUITY RETURN CORRELATION")
    print(correlation.round(3).to_string())

    logging.info("EDA completed successfully.")


if __name__ == "__main__":
    main()
