import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

PRICE_PATH = ROOT_DIR / "data" / "raw" / "prices.parquet"

REPORT_DIR = ROOT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRADING_DAYS = 252

# Assumption for this educational portfolio.
# This is not a live or current market risk-free rate.
RISK_FREE_RATE = 0.04

MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

TICKERS = [
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "NVDA",
    "KO",
    "CAT",
    "HD",
]


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD PRICE DATA
# --------------------------------------------------

def load_prices():
    logger.info("Loading historical price data.")

    if not PRICE_PATH.exists():
        raise FileNotFoundError(
            f"Price file not found: {PRICE_PATH}"
        )

    df = pd.read_parquet(PRICE_PATH)

    logger.info(
        "Loaded price data with shape: %s",
        df.shape,
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Standardize common column names.
    lower_map = {
        str(column).lower().replace(" ", "_"): column
        for column in df.columns
    }

    date_column = None

    for candidate in ["date", "datetime"]:
        if candidate in lower_map:
            date_column = lower_map[candidate]
            break

    ticker_column = None

    for candidate in ["ticker", "symbol"]:
        if candidate in lower_map:
            ticker_column = lower_map[candidate]
            break

    if date_column is None:
        raise ValueError(
            "Could not identify the date column. "
            f"Available columns: {list(df.columns)}"
        )

    if ticker_column is None:
        raise ValueError(
            "Could not identify the ticker column. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    df[ticker_column] = (
        df[ticker_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Prefer adjusted close when available.
    price_column = None

    candidates = [
        "adj_close",
        "adjusted_close",
        "close",
    ]

    for candidate in candidates:
        if candidate in lower_map:
            price_column = lower_map[candidate]
            break

    if price_column is None:
        raise ValueError(
            "Could not identify a price column. "
            f"Available columns: {list(df.columns)}"
        )

    df[price_column] = pd.to_numeric(
        df[price_column],
        errors="coerce",
    )

    df = df[
        [
            date_column,
            ticker_column,
            price_column,
        ]
    ].copy()

    df = df.rename(
        columns={
            date_column: "Date",
            ticker_column: "Ticker",
            price_column: "Price",
        }
    )

    df = df[
        df["Ticker"].isin(TICKERS)
    ]

    df = df.dropna(
        subset=["Date", "Ticker", "Price"]
    )

    df = df[
        df["Price"] > 0
    ]

    if df.empty:
        raise ValueError(
            "No valid price observations found."
        )

    if df.duplicated(
        subset=["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker records found."
        )

    prices = df.pivot(
        index="Date",
        columns="Ticker",
        values="Price",
    )

    prices = prices.sort_index()

    missing_tickers = [
        ticker
        for ticker in TICKERS
        if ticker not in prices.columns
    ]

    if missing_tickers:
        raise ValueError(
            f"Missing ticker data: {missing_tickers}"
        )

    prices = prices[TICKERS]

    logger.info(
        "Price matrix shape: %s",
        prices.shape,
    )

    logger.info(
        "Price date range: %s to %s",
        prices.index.min(),
        prices.index.max(),
    )

    # Keep dates where all selected assets have observations.
    prices = prices.dropna(
        how="any"
    )

    if len(prices) < 100:
        raise ValueError(
            "Insufficient complete historical observations."
        )

    logger.info(
        "Complete price matrix shape: %s",
        prices.shape,
    )

    return prices


# --------------------------------------------------
# CALCULATE RETURNS
# --------------------------------------------------

def calculate_returns(prices):
    logger.info("Calculating daily returns.")

    daily_returns = prices.pct_change()

    daily_returns = daily_returns.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    daily_returns = daily_returns.dropna(
        how="any"
    )

    if daily_returns.empty:
        raise ValueError(
            "No valid daily returns available."
        )

    logger.info(
        "Daily returns shape: %s",
        daily_returns.shape,
    )

    return daily_returns


# --------------------------------------------------
# PORTFOLIO STATISTICS
# --------------------------------------------------

def portfolio_statistics(
    weights,
    mean_returns,
    covariance,
):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    expected_return = float(
        np.dot(
            weights,
            mean_returns,
        )
        * TRADING_DAYS
    )

    variance = float(
        weights.T
        @ covariance
        @ weights
        * TRADING_DAYS
    )

    volatility = float(
        np.sqrt(
            max(variance, 0.0)
        )
    )

    if volatility > 0:
        sharpe_ratio = (
            expected_return
            - RISK_FREE_RATE
        ) / volatility
    else:
        sharpe_ratio = np.nan

    return {
        "Expected_Annual_Return": expected_return,
        "Annual_Volatility": volatility,
        "Sharpe_Ratio": sharpe_ratio,
    }


# --------------------------------------------------
# OBJECTIVE FUNCTIONS
# --------------------------------------------------

def minimum_volatility_objective(
    weights,
    covariance,
):
    variance = (
        weights.T
        @ covariance
        @ weights
    )

    return float(
        max(variance, 0.0)
    )


def negative_sharpe_objective(
    weights,
    mean_returns,
    covariance,
):
    stats = portfolio_statistics(
        weights,
        mean_returns,
        covariance,
    )

    sharpe = stats["Sharpe_Ratio"]

    if not np.isfinite(sharpe):
        return 1e6

    return -sharpe


# --------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------

def optimize_portfolio(
    mean_returns,
    covariance,
    objective_name,
):
    asset_count = len(
        mean_returns
    )

    initial_weights = np.full(
        asset_count,
        1.0 / asset_count,
    )

    bounds = [
        (MIN_WEIGHT, MAX_WEIGHT)
        for _ in range(asset_count)
    ]

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
        }
    ]

    if objective_name == "minimum_volatility":

        objective = lambda weights: (
            minimum_volatility_objective(
                weights,
                covariance,
            )
        )

    elif objective_name == "maximum_sharpe":

        objective = lambda weights: (
            negative_sharpe_objective(
                weights,
                mean_returns,
                covariance,
            )
        )

    else:
        raise ValueError(
            f"Unsupported objective: {objective_name}"
        )

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-10,
        },
    )

    if not result.success:
        logger.warning(
            "%s optimization warning: %s",
            objective_name,
            result.message,
        )

    weights = np.asarray(
        result.x,
        dtype=float,
    )

    # Remove numerical noise and normalize.
    weights = np.clip(
        weights,
        0.0,
        1.0,
    )

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "Optimization produced invalid weights."
        )

    weights = weights / total

    return weights


# --------------------------------------------------
# BUILD PORTFOLIOS
# --------------------------------------------------

def build_portfolios(daily_returns):
    logger.info(
        "Estimating historical return and covariance."
    )

    mean_returns = (
        daily_returns.mean()
        .to_numpy()
    )

    covariance = (
        daily_returns.cov()
        .to_numpy()
    )

    # Equal-weight portfolio
    equal_weights = np.full(
        len(TICKERS),
        1.0 / len(TICKERS),
    )

    # Minimum volatility
    min_vol_weights = optimize_portfolio(
        mean_returns,
        covariance,
        "minimum_volatility",
    )

    # Maximum Sharpe
    max_sharpe_weights = optimize_portfolio(
        mean_returns,
        covariance,
        "maximum_sharpe",
    )

    portfolios = {
        "Equal Weight": equal_weights,
        "Minimum Volatility": min_vol_weights,
        "Maximum Sharpe": max_sharpe_weights,
    }

    return portfolios, mean_returns, covariance


# --------------------------------------------------
# CREATE WEIGHT REPORT
# --------------------------------------------------

def create_weight_report(
    portfolios,
    mean_returns,
    covariance,
):
    logger.info(
        "Generating portfolio allocation report."
    )

    rows = []

    for portfolio_name, weights in portfolios.items():

        stats = portfolio_statistics(
            weights,
            mean_returns,
            covariance,
        )

        for ticker, weight in zip(
            TICKERS,
            weights,
        ):

            rows.append(
                {
                    "Portfolio": portfolio_name,
                    "Ticker": ticker,
                    "Weight": float(weight),
                    "Weight_Percent": float(
                        weight * 100
                    ),
                    "Expected_Annual_Return": (
                        stats["Expected_Annual_Return"]
                    ),
                    "Annual_Volatility": (
                        stats["Annual_Volatility"]
                    ),
                    "Sharpe_Ratio": (
                        stats["Sharpe_Ratio"]
                    ),
                }
            )

    report = pd.DataFrame(rows)

    output_path = (
        REPORT_DIR
        / "portfolio_allocation_weights.csv"
    )

    report.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved portfolio allocation report: %s",
        output_path,
    )

    print("\n" + "=" * 80)
    print("PORTFOLIO ALLOCATION WEIGHTS")
    print("=" * 80)

    for portfolio_name in portfolios:

        subset = report[
            report["Portfolio"] == portfolio_name
        ].copy()

        subset = subset.sort_values(
            "Weight",
            ascending=False,
        )

        print(f"\n{portfolio_name}")

        print(
            subset[
                [
                    "Ticker",
                    "Weight_Percent",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )

        stats = subset.iloc[0]

        print(
            f"Expected annual return: "
            f"{stats['Expected_Annual_Return']:.4f}"
        )

        print(
            f"Annual volatility: "
            f"{stats['Annual_Volatility']:.4f}"
        )

        print(
            f"Sharpe ratio: "
            f"{stats['Sharpe_Ratio']:.4f}"
        )

        print(
            f"Weight total: "
            f"{subset['Weight'].sum():.6f}"
        )

    return report


# --------------------------------------------------
# SUMMARY REPORT
# --------------------------------------------------

def create_summary_report(
    portfolios,
    mean_returns,
    covariance,
):
    rows = []

    for portfolio_name, weights in portfolios.items():

        stats = portfolio_statistics(
            weights,
            mean_returns,
            covariance,
        )

        rows.append(
            {
                "Portfolio": portfolio_name,
                **stats,
                "Weight_Total": float(
                    weights.sum()
                ),
                "Assets": len(weights),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    output_path = (
        REPORT_DIR
        / "portfolio_optimization_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved portfolio summary: %s",
        output_path,
    )

    print("\n" + "=" * 80)
    print("PORTFOLIO COMPARISON")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return summary


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    logger.info(
        "Starting portfolio optimization."
    )

    prices = load_prices()

    daily_returns = calculate_returns(
        prices
    )

    portfolios, mean_returns, covariance = (
        build_portfolios(
            daily_returns
        )
    )

    weight_report = create_weight_report(
        portfolios,
        mean_returns,
        covariance,
    )

    summary_report = create_summary_report(
        portfolios,
        mean_returns,
        covariance,
    )

    logger.info(
        "Portfolio optimization completed."
    )

    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION COMPLETED")
    print("=" * 80)

    print(
        f"Historical observations: {len(daily_returns)}"
    )

    print(
        f"Assets: {len(TICKERS)}"
    )

    print(
        f"Weight report: "
        f"{REPORT_DIR / 'portfolio_allocation_weights.csv'}"
    )

    print(
        f"Summary report: "
        f"{REPORT_DIR / 'portfolio_optimization_summary.csv'}"
    )


if __name__ == "__main__":
    main()