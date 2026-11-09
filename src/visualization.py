
"""Visual exploratory data analysis for Northgate AI Stock Predictor."""

from __future__ import annotations

import logging

import matplotlib

# Use a non-interactive backend suitable for Windows and file output.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load combined features and equity return correlations."""

    config = load_config()

    processed_dir = project_path(config["paths"]["processed_data"])
    reports_dir = project_path(config["paths"]["reports"])

    combined_path = processed_dir / "combined_features.parquet"
    correlation_path = reports_dir / "equity_return_correlation.csv"

    if not combined_path.exists():
        raise FileNotFoundError(
            f"Combined dataset not found: {combined_path}"
        )

    if not correlation_path.exists():
        raise FileNotFoundError(
            f"Correlation file not found: {correlation_path}"
        )

    data = pd.read_parquet(combined_path)
    correlation = pd.read_csv(correlation_path, index_col=0)

    data["Date"] = pd.to_datetime(data["Date"])

    logging.info("Loaded combined dataset: %s", data.shape)
    logging.info("Loaded correlation matrix: %s", correlation.shape)

    return data, correlation


def create_output_directory() -> None:
    """Create the configured figures directory."""

    config = load_config()
    figures_dir = project_path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Figure output directory: %s", figures_dir)


def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure."""

    config = load_config()
    figures_dir = project_path(config["paths"]["figures"])

    output_path = figures_dir / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info("Saved figure: %s", output_path)


def plot_normalized_stock_performance(data: pd.DataFrame) -> None:
    """Compare adjusted prices normalized to 100 at the beginning."""

    config = load_config()
    equities = config["data"]["equity_universe"]

    stock_data = data[data["Ticker"].isin(equities)].copy()
    stock_data = stock_data.sort_values(["Ticker", "Date"])

    plt.figure(figsize=(15, 8))

    for ticker, group in stock_data.groupby("Ticker"):
        group = group.sort_values("Date")

        first_price = group["Adj Close"].iloc[0]

        normalized_price = (
            group["Adj Close"] / first_price
        ) * 100

        plt.plot(
            group["Date"],
            normalized_price,
            label=ticker,
            linewidth=1.5,
        )

    plt.title("Normalized Stock Performance (Base = 100)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Adjusted Close")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)

    save_figure("normalized_stock_performance.png")


def plot_daily_return_distributions(data: pd.DataFrame) -> None:
    """Plot daily simple-return distributions for the 10 equities."""

    config = load_config()
    equities = config["data"]["equity_universe"]

    stock_data = data[data["Ticker"].isin(equities)].copy()

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(14, 18),
    )

    axes = axes.flatten()

    for index, ticker in enumerate(equities):
        group = stock_data[stock_data["Ticker"] == ticker]
        returns = group["Daily_Return"].dropna()

        axes[index].hist(
            returns,
            bins=60,
            density=True,
            alpha=0.75,
        )

        axes[index].set_title(f"{ticker} Daily Returns")
        axes[index].set_xlabel("Daily Return")
        axes[index].set_ylabel("Density")
        axes[index].grid(True, alpha=0.25)

    plt.suptitle(
        "Daily Return Distributions",
        fontsize=16,
    )

    save_figure("daily_return_distributions.png")


def plot_annualized_volatility(data: pd.DataFrame) -> None:
    """Compare annualized historical volatility across equities."""

    config = load_config()
    equities = config["data"]["equity_universe"]

    stock_data = data[data["Ticker"].isin(equities)].copy()

    volatility = (
        stock_data.groupby("Ticker")["Daily_Return"]
        .std()
        * np.sqrt(252)
    )

    volatility = volatility.sort_values(ascending=False)

    plt.figure(figsize=(12, 7))

    plt.bar(
        volatility.index,
        volatility.values,
    )

    plt.title("Annualized Historical Volatility")
    plt.xlabel("Ticker")
    plt.ylabel("Annualized Volatility")
    plt.grid(axis="y", alpha=0.3)

    save_figure("annualized_volatility.png")


def plot_maximum_drawdown(data: pd.DataFrame) -> None:
    """Compare maximum historical drawdown across equities."""

    config = load_config()
    equities = config["data"]["equity_universe"]

    stock_data = data[data["Ticker"].isin(equities)].copy()
    stock_data = stock_data.sort_values(["Ticker", "Date"])

    drawdowns = {}

    for ticker, group in stock_data.groupby("Ticker"):
        group = group.sort_values("Date")

        adjusted_close = group["Adj Close"]

        running_max = adjusted_close.cummax()

        drawdown = (
            adjusted_close / running_max
        ) - 1

        drawdowns[ticker] = drawdown.min()

    drawdown_series = pd.Series(drawdowns)
    drawdown_series = drawdown_series.sort_values()

    plt.figure(figsize=(12, 7))

    plt.bar(
        drawdown_series.index,
        drawdown_series.values,
    )

    plt.title("Maximum Historical Drawdown")
    plt.xlabel("Ticker")
    plt.ylabel("Maximum Drawdown")
    plt.grid(axis="y", alpha=0.3)

    save_figure("maximum_drawdown.png")


def plot_correlation_heatmap(correlation: pd.DataFrame) -> None:
    """Visualize the equity daily-return correlation matrix."""

    correlation = correlation.astype(float)

    fig, ax = plt.subplots(figsize=(12, 10))

    image = ax.imshow(
        correlation.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.index)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(correlation.index)

    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            value = correlation.iloc[row, column]

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title("Equity Daily Return Correlation")

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    save_figure("equity_return_correlation_heatmap.png")


def plot_technical_indicators(data: pd.DataFrame) -> None:
    """Plot adjusted price, moving averages, and RSI for each equity."""

    config = load_config()
    equities = config["data"]["equity_universe"]

    stock_data = data[data["Ticker"].isin(equities)].copy()

    for ticker in equities:
        group = stock_data[
            stock_data["Ticker"] == ticker
        ].sort_values("Date")

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(15, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        axes[0].plot(
            group["Date"],
            group["Adj Close"],
            label="Adjusted Close",
            linewidth=1.2,
        )

        axes[0].plot(
            group["Date"],
            group["SMA_20"],
            label="SMA 20",
            linewidth=1,
        )

        axes[0].plot(
            group["Date"],
            group["SMA_50"],
            label="SMA 50",
            linewidth=1,
        )

        axes[0].plot(
            group["Date"],
            group["SMA_200"],
            label="SMA 200",
            linewidth=1,
        )

        axes[0].set_title(
            f"{ticker} Price and Moving Averages"
        )

        axes[0].set_ylabel("Adjusted Close")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(
            group["Date"],
            group["RSI_14"],
            label="RSI 14",
            linewidth=1,
        )

        axes[1].axhline(
            70,
            linestyle="--",
            linewidth=1,
            label="70",
        )

        axes[1].axhline(
            30,
            linestyle="--",
            linewidth=1,
            label="30",
        )

        axes[1].set_title("Relative Strength Index")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("RSI")
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)

        save_figure(f"{ticker}_technical_indicators.png")


def main() -> None:
    configure_logging()

    logging.info("Starting visual exploratory data analysis.")

    create_output_directory()

    data, correlation = load_data()

    plot_normalized_stock_performance(data)
    plot_daily_return_distributions(data)
    plot_annualized_volatility(data)
    plot_maximum_drawdown(data)
    plot_correlation_heatmap(correlation)
    plot_technical_indicators(data)

    logging.info("Visual EDA completed successfully.")


if __name__ == "__main__":
    main()
