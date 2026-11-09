
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

PRICE_PATH = ROOT_DIR / "data" / "raw" / "prices.parquet"
BENCHMARK_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "benchmark.parquet"
)

REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

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

BENCHMARK = "^GSPC"

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"

TEST_START = "2024-01-01"
TEST_END = "2026-10-31"

TRADING_DAYS = 252

RISK_FREE_RATE = 0.04

# A fixed 4% annual risk-free rate is used for this educational
# comparison. It is an assumption, not a current market rate.

INITIAL_CAPITAL = 100000.0

# Include estimated one-way transaction costs.
# 0.001 = 0.10% per unit of traded portfolio weight.
TRANSACTION_COST = 0.001

# Maximum allocation to any one asset.
MAX_WEIGHT = 0.30

# This constraint applies to optimized portfolios.
# Equal Weight is retained as the simple baseline.


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)



# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_price_data():

    logger.info("Loading historical price data.")

    if not PRICE_PATH.exists():
        raise FileNotFoundError(
            f"Price file not found: {PRICE_PATH}"
        )

    df = pd.read_parquet(PRICE_PATH)

    logger.info(
        "Loaded price dataset shape: %s",
        df.shape,
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    lower_map = {
        str(column).lower().replace(" ", "_"): column
        for column in df.columns
    }

    # --------------------------------------------------
    # IDENTIFY REQUIRED COLUMNS
    # --------------------------------------------------

    date_column = next(
        (
            lower_map[c]
            for c in ["date", "datetime"]
            if c in lower_map
        ),
        None,
    )

    ticker_column = next(
        (
            lower_map[c]
            for c in ["ticker", "symbol"]
            if c in lower_map
        ),
        None,
    )

    price_column = next(
        (
            lower_map[c]
            for c in [
                "adj_close",
                "adjusted_close",
                "close",
            ]
            if c in lower_map
        ),
        None,
    )

    if date_column is None:
        raise ValueError(
            f"Date column not found: {list(df.columns)}"
        )

    if ticker_column is None:
        raise ValueError(
            f"Ticker column not found: {list(df.columns)}"
        )

    if price_column is None:
        raise ValueError(
            f"Price column not found: {list(df.columns)}"
        )

    # --------------------------------------------------
    # STANDARDIZE STOCK DATA
    # --------------------------------------------------

    df = df[
        [
            date_column,
            ticker_column,
            price_column,
        ]
    ].copy()

    df.columns = [
        "Date",
        "Ticker",
        "Price",
    ]

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date", "Ticker", "Price"]
    )

    df = df[df["Price"] > 0]

    # --------------------------------------------------
    # LOAD BENCHMARK BEFORE PIVOTING
    # --------------------------------------------------

    if not BENCHMARK_PATH.exists():

        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_PATH}"
        )

    logger.info(
        "Loading separate S&P 500 benchmark data."
    )

    benchmark_df = pd.read_parquet(
        BENCHMARK_PATH
    )

    benchmark_df.columns = [
        str(column).strip()
        for column in benchmark_df.columns
    ]

    benchmark_columns = {
        str(column).lower().replace(" ", "_"): column
        for column in benchmark_df.columns
    }

    required_benchmark_columns = [
        "date",
        "ticker",
        "price",
    ]

    missing_columns = [
        column
        for column in required_benchmark_columns
        if column not in benchmark_columns
    ]

    if missing_columns:

        raise ValueError(
            f"Benchmark columns missing: {missing_columns}"
        )

    benchmark_df = benchmark_df[
        [
            benchmark_columns["date"],
            benchmark_columns["ticker"],
            benchmark_columns["price"],
        ]
    ].copy()

    benchmark_df.columns = [
        "Date",
        "Ticker",
        "Price",
    ]

    benchmark_df["Date"] = pd.to_datetime(
        benchmark_df["Date"]
    )

    benchmark_df["Ticker"] = (
        benchmark_df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    benchmark_df["Price"] = pd.to_numeric(
        benchmark_df["Price"],
        errors="coerce",
    )

    benchmark_df = benchmark_df.dropna(
        subset=["Date", "Ticker", "Price"]
    )

    benchmark_df = benchmark_df[
        benchmark_df["Ticker"] == BENCHMARK
    ]

    benchmark_df = benchmark_df[
        benchmark_df["Price"] > 0
    ]

    if benchmark_df.empty:

        raise ValueError(
            "No valid S&P 500 benchmark records found."
        )

    if benchmark_df.duplicated(
        subset=["Date", "Ticker"]
    ).any():

        raise ValueError(
            "Duplicate benchmark Date-Ticker records found."
        )


    # --------------------------------------------------
    # MERGE STOCKS AND BENCHMARK
    # --------------------------------------------------

    # Remove benchmark records from the original dataset.
    # We will use the separately downloaded benchmark file.

    df = df[
        df["Ticker"] != BENCHMARK
    ].copy()

    logger.info(
        "Removed existing benchmark records from stock dataset."
    )

    # Append the separately downloaded benchmark.
    df = pd.concat(
        [
            df,
            benchmark_df,
        ],
        ignore_index=True,
    )

    logger.info(
        "Merged benchmark observations: %s",
        len(benchmark_df),
    )

    # --------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------

    df = df.drop_duplicates(
        subset=["Date", "Ticker"],
        keep="last",
    )

    logger.info(
        "Duplicate Date-Ticker records removed."
    )


    # --------------------------------------------------
    # FILTER STOCKS AND BENCHMARK
    # --------------------------------------------------

    requested_tickers = (
        TICKERS + [BENCHMARK]
    )

    df = df[
        df["Ticker"].isin(
            requested_tickers
        )
    ]

    if df.empty:

        raise ValueError(
            "No requested stock or benchmark records found."
        )

    if df.duplicated(
        subset=["Date", "Ticker"]
    ).any():

        raise ValueError(
            "Duplicate Date-Ticker records found."
        )

    # --------------------------------------------------
    # CREATE PRICE MATRIX
    # --------------------------------------------------

    prices = df.pivot(
        index="Date",
        columns="Ticker",
        values="Price",
    )

    prices = prices.sort_index()

    # --------------------------------------------------
    # VALIDATE STOCK TICKERS
    # --------------------------------------------------

    available_tickers = [
        ticker
        for ticker in TICKERS
        if ticker in prices.columns
    ]

    missing_tickers = [
        ticker
        for ticker in TICKERS
        if ticker not in prices.columns
    ]

    if missing_tickers:

        raise ValueError(
            f"Missing stock tickers: {missing_tickers}"
        )

    # Keep stock prices separate from benchmark.
    stock_prices = prices[
        available_tickers
    ].copy()

    # --------------------------------------------------
    # VALIDATE BENCHMARK
    # --------------------------------------------------

    benchmark_available = (
        BENCHMARK in prices.columns
    )

    if benchmark_available:

        logger.info(
            "S&P 500 benchmark successfully integrated."
        )

        logger.info(
            "Benchmark dates: %s to %s",
            prices[BENCHMARK].dropna().index.min(),
            prices[BENCHMARK].dropna().index.max(),
        )

    else:

        raise ValueError(
            "S&P 500 benchmark missing after merge."
        )

    logger.info(
        "Stock price matrix shape: %s",
        stock_prices.shape,
    )

    logger.info(
        "Available price dates: %s to %s",
        stock_prices.index.min(),
        stock_prices.index.max(),
    )

    return prices, benchmark_available

# --------------------------------------------------
# SPLIT TRAIN AND TEST
# --------------------------------------------------

def split_data(prices):

    logger.info(
        "Splitting optimization and test periods."
    )

    train_prices = prices.loc[
        TRAIN_START:TRAIN_END
    ].copy()

    test_prices = prices.loc[
        TEST_START:TEST_END
    ].copy()

    # The optimized stock portfolios require complete observations
    # for all ten stocks on a given date.
    train_prices = train_prices[
        TICKERS
    ].dropna(
        how="any"
    )

    test_prices = test_prices[
        TICKERS
    ].dropna(
        how="any"
    )

    if len(train_prices) < 100:
        raise ValueError(
            "Insufficient training observations."
        )

    if len(test_prices) < 100:
        raise ValueError(
            "Insufficient test observations."
        )

    logger.info(
        "Training prices: %s rows, %s to %s",
        len(train_prices),
        train_prices.index.min(),
        train_prices.index.max(),
    )

    logger.info(
        "Test prices: %s rows, %s to %s",
        len(test_prices),
        test_prices.index.min(),
        test_prices.index.max(),
    )

    if train_prices.index.max() >= test_prices.index.min():
        raise ValueError(
            "Training and test periods overlap."
        )

    return train_prices, test_prices


# --------------------------------------------------
# TRAINING RETURNS
# --------------------------------------------------

def calculate_training_returns(train_prices):

    logger.info(
        "Calculating training-period returns."
    )

    returns = train_prices.pct_change()

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    returns = returns.dropna(
        how="any"
    )

    if returns.empty:
        raise ValueError(
            "Training returns are empty."
        )

    return returns


# --------------------------------------------------
# PORTFOLIO METRICS
# --------------------------------------------------

def calculate_portfolio_statistics(
    weights,
    mean_returns,
    covariance,
):

    weights = np.asarray(
        weights,
        dtype=float,
    )

    annual_return = float(
        np.dot(
            weights,
            mean_returns,
        ) * TRADING_DAYS
    )

    annual_variance = float(
        weights.T
        @ covariance
        @ weights
        * TRADING_DAYS
    )

    annual_volatility = float(
        np.sqrt(
            max(annual_variance, 0.0)
        )
    )

    if annual_volatility > 0:

        sharpe = (
            annual_return
            - RISK_FREE_RATE
        ) / annual_volatility

    else:
        sharpe = np.nan

    return {
        "Expected_Annual_Return": annual_return,
        "Annual_Volatility": annual_volatility,
        "Sharpe_Ratio": sharpe,
    }


# --------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------

def optimize_weights(
    mean_returns,
    covariance,
    method,
):

    asset_count = len(
        mean_returns
    )

    initial_weights = np.full(
        asset_count,
        1.0 / asset_count,
    )

    bounds = [
        (0.0, MAX_WEIGHT)
        for _ in range(asset_count)
    ]

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: (
                np.sum(weights) - 1.0
            ),
        }
    ]

    if method == "minimum_volatility":

        objective = lambda weights: (
            weights.T
            @ covariance
            @ weights
        )

    elif method == "maximum_sharpe":

        def objective(weights):

            stats = calculate_portfolio_statistics(
                weights,
                mean_returns,
                covariance,
            )

            sharpe = stats["Sharpe_Ratio"]

            if not np.isfinite(sharpe):
                return 1e6

            return -sharpe

    else:
        raise ValueError(
            f"Unsupported optimization method: {method}"
        )

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 3000,
            "ftol": 1e-10,
        },
    )

    if not result.success:

        logger.warning(
            "%s optimization did not fully converge: %s",
            method,
            result.message,
        )

    weights = np.clip(
        np.asarray(result.x, dtype=float),
        0.0,
        MAX_WEIGHT,
    )

    if weights.sum() <= 0:

        raise ValueError(
            "Optimizer returned invalid weights."
        )

    weights = weights / weights.sum()

    return weights


# --------------------------------------------------
# FIT PORTFOLIOS USING TRAINING DATA ONLY
# --------------------------------------------------

def fit_portfolios(train_returns):

    logger.info(
        "Optimizing portfolios using training data only."
    )

    mean_returns = (
        train_returns.mean()
        .to_numpy()
    )

    covariance = (
        train_returns.cov()
        .to_numpy()
    )

    equal_weight = np.full(
        len(TICKERS),
        1.0 / len(TICKERS),
    )

    minimum_volatility = optimize_weights(
        mean_returns,
        covariance,
        "minimum_volatility",
    )

    maximum_sharpe = optimize_weights(
        mean_returns,
        covariance,
        "maximum_sharpe",
    )

    portfolios = {
        "Equal Weight": equal_weight,
        "Minimum Volatility": minimum_volatility,
        "Maximum Sharpe": maximum_sharpe,
    }

    logger.info(
        "Training-period portfolio optimization completed."
    )

    for name, weights in portfolios.items():

        stats = calculate_portfolio_statistics(
            weights,
            mean_returns,
            covariance,
        )

        logger.info(
            "%s | Train annual return=%.4f | "
            "volatility=%.4f | Sharpe=%.4f",
            name,
            stats["Expected_Annual_Return"],
            stats["Annual_Volatility"],
            stats["Sharpe_Ratio"],
        )

    return portfolios


# --------------------------------------------------
# BACKTEST
# --------------------------------------------------

def backtest_portfolio(
    test_returns,
    weights,
    portfolio_name,
):

    logger.info(
        "Backtesting %s.",
        portfolio_name,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    # Portfolio return at date t uses asset returns at t.
    # Weights are fixed using training-period data only.
    gross_returns = (
        test_returns[TICKERS]
        @ weights
    )

    # One initial allocation from cash into the portfolio.
    # The cost is applied once at the start.
    initial_cost = TRANSACTION_COST

    net_returns = gross_returns.copy()

    if len(net_returns) > 0:

        net_returns.iloc[0] = (
            (1.0 - initial_cost)
            * (1.0 + gross_returns.iloc[0])
            - 1.0
        )

    cumulative_value = (
        INITIAL_CAPITAL
        * (1.0 + net_returns)
        .cumprod()
    )

    result = pd.DataFrame(
        {
            "Date": test_returns.index,
            "Portfolio": portfolio_name,
            "Daily_Return": net_returns.to_numpy(),
            "Gross_Daily_Return": gross_returns.to_numpy(),
            "Portfolio_Value": cumulative_value.to_numpy(),
        }
    )

    return result


# --------------------------------------------------
# BENCHMARK BACKTEST
# --------------------------------------------------

def backtest_benchmark(
    benchmark_prices,
):

    logger.info(
        "Calculating benchmark performance."
    )

    benchmark_returns = (
        benchmark_prices
        .pct_change()
        .dropna()
    )

    benchmark_returns = benchmark_returns.loc[
        TEST_START:TEST_END
    ]

    if benchmark_returns.empty:
        raise ValueError(
            "No benchmark returns available for test period."
        )

    benchmark_net_returns = benchmark_returns.copy()

    if len(benchmark_net_returns) > 0:

        benchmark_net_returns.iloc[0] = (
            (1.0 - TRANSACTION_COST)
            * (1.0 + benchmark_returns.iloc[0])
            - 1.0
        )

    benchmark_value = (
        INITIAL_CAPITAL
        * (1.0 + benchmark_net_returns)
        .cumprod()
    )

    result = pd.DataFrame(
        {
            "Date": benchmark_net_returns.index,
            "Portfolio": "S&P 500",
            "Daily_Return": benchmark_net_returns.to_numpy(),
            "Gross_Daily_Return": benchmark_returns.to_numpy(),
            "Portfolio_Value": benchmark_value.to_numpy(),
        }
    )

    return result


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def evaluate_strategy(backtest_df):

    daily_returns = (
        backtest_df["Daily_Return"]
        .astype(float)
    )

    portfolio_values = (
        backtest_df["Portfolio_Value"]
        .astype(float)
    )

    observation_count = len(
        daily_returns
    )

    if observation_count == 0:
        raise ValueError(
            "Cannot evaluate empty backtest."
        )

    total_return = (
        portfolio_values.iloc[-1]
        / INITIAL_CAPITAL
        - 1.0
    )

    annualized_return = (
        (1.0 + total_return)
        ** (TRADING_DAYS / observation_count)
        - 1.0
    )

    annualized_volatility = (
        daily_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    if annualized_volatility > 0:

        sharpe_ratio = (
            annualized_return
            - RISK_FREE_RATE
        ) / annualized_volatility

    else:
        sharpe_ratio = np.nan

    running_max = (
        portfolio_values
        .cummax()
    )

    drawdown = (
        portfolio_values
        / running_max
        - 1.0
    )

    maximum_drawdown = drawdown.min()

    final_value = portfolio_values.iloc[-1]

    return {
        "Portfolio": backtest_df["Portfolio"].iloc[0],
        "Observations": observation_count,
        "Initial_Capital": INITIAL_CAPITAL,
        "Final_Portfolio_Value": final_value,
        "Total_Return": total_return,
        "Annualized_Return": annualized_return,
        "Annualized_Volatility": annualized_volatility,
        "Sharpe_Ratio": sharpe_ratio,
        "Maximum_Drawdown": maximum_drawdown,
    }


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_backtest_results(
    backtest_results,
    portfolio_metrics,
):

    logger.info(
        "Saving backtest results."
    )

    all_daily_results = pd.concat(
        backtest_results,
        ignore_index=True,
    )

    daily_path = (
        REPORT_DIR
        / "portfolio_backtest_daily.csv"
    )

    all_daily_results.to_csv(
        daily_path,
        index=False,
    )

    metrics_df = pd.DataFrame(
        portfolio_metrics
    )

    metrics_path = (
        REPORT_DIR
        / "portfolio_backtest_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    logger.info(
        "Saved daily backtest: %s",
        daily_path,
    )

    logger.info(
        "Saved backtest metrics: %s",
        metrics_path,
    )

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE BACKTEST RESULTS")
    print("=" * 80)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return all_daily_results, metrics_df


# --------------------------------------------------
# PLOT CUMULATIVE PERFORMANCE
# --------------------------------------------------

def plot_cumulative_performance(
    all_daily_results,
):

    logger.info(
        "Creating cumulative portfolio performance chart."
    )

    plt.figure(
        figsize=(13, 7)
    )

    for portfolio_name, group in (
        all_daily_results.groupby("Portfolio")
    ):

        group = group.sort_values(
            "Date"
        )

        plt.plot(
            group["Date"],
            group["Portfolio_Value"],
            label=portfolio_name,
        )

    plt.title(
        "Out-of-Sample Portfolio Performance"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "portfolio_backtest_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved performance chart: %s",
        output_path,
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    logger.info(
        "Starting out-of-sample portfolio backtesting."
    )

    prices, benchmark_available = (
        load_price_data()
    )

    train_prices, test_prices = (
        split_data(prices)
    )

    train_returns = (
        calculate_training_returns(
            train_prices
        )
    )

    # Optimize using training-period returns only.
    portfolios = fit_portfolios(
        train_returns
    )

    # Calculate test-period returns.
    test_returns = (
        test_prices
        .pct_change()
        .dropna(
            how="any"
        )
    )

    if test_returns.empty:
        raise ValueError(
            "Test-period returns are empty."
        )

    backtest_results = []
    portfolio_metrics = []

    # Backtest each fixed portfolio.
    for portfolio_name, weights in (
        portfolios.items()
    ):

        daily_result = (
            backtest_portfolio(
                test_returns,
                weights,
                portfolio_name,
            )
        )

        metrics = evaluate_strategy(
            daily_result
        )

        backtest_results.append(
            daily_result
        )

        portfolio_metrics.append(
            metrics
        )

    # Benchmark comparison.
    if benchmark_available:

        benchmark_prices = prices[
            BENCHMARK
        ].dropna()

        benchmark_result = (
            backtest_benchmark(
                benchmark_prices
            )
        )

        benchmark_metrics = (
            evaluate_strategy(
                benchmark_result
            )
        )

        backtest_results.append(
            benchmark_result
        )

        portfolio_metrics.append(
            benchmark_metrics
        )

    # Save daily and summary results.
    all_daily_results, metrics_df = (
        save_backtest_results(
            backtest_results,
            portfolio_metrics,
        )
    )

    # Plot results.
    plot_cumulative_performance(
        all_daily_results
    )

    logger.info(
        "Out-of-sample backtesting completed."
    )

    print("\n" + "=" * 80)
    print("BACKTESTING COMPLETED")
    print("=" * 80)

    print(
        f"Training period: {TRAIN_START} to {TRAIN_END}"
    )

    print(
        f"Testing period: {TEST_START} to {TEST_END}"
    )

    print(
        f"Daily results: "
        f"{REPORT_DIR / 'portfolio_backtest_daily.csv'}"
    )

    print(
        f"Metrics: "
        f"{REPORT_DIR / 'portfolio_backtest_metrics.csv'}"
    )

    print(
        f"Chart: "
        f"{FIGURE_DIR / 'portfolio_backtest_performance.png'}"
    )


if __name__ == "__main__":
    main()
