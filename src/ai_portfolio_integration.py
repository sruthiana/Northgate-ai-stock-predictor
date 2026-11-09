
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

PORTFOLIO_DAILY = (
    REPORT_DIR / "portfolio_backtest_daily.csv"
)

STRATEGY_DAILY = (
    REPORT_DIR / "strategy_comparison_daily.csv"
)

PORTFOLIO_METRICS = (
    REPORT_DIR / "portfolio_backtest_metrics.csv"
)

STRATEGY_METRICS = (
    REPORT_DIR / "strategy_comparison_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100000.0
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_files():

    required_files = [
        PORTFOLIO_DAILY,
        STRATEGY_DAILY,
        PORTFOLIO_METRICS,
        STRATEGY_METRICS,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Required integration files are missing:\n"
            + "\n".join(missing)
        )

    logger.info(
        "All required input files are available."
    )


# ============================================================
# LOAD PORTFOLIO BACKTEST
# ============================================================

def load_portfolio_backtest():

    logger.info(
        "Loading optimized portfolio backtest."
    )

    df = pd.read_csv(
        PORTFOLIO_DAILY
    )

    required_columns = [
        "Date",
        "Portfolio",
        "Daily_Return",
        "Portfolio_Value",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Portfolio backtest columns missing: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Daily_Return"] = pd.to_numeric(
        df["Daily_Return"],
        errors="coerce",
    )

    df["Portfolio_Value"] = pd.to_numeric(
        df["Portfolio_Value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Portfolio",
            "Daily_Return",
            "Portfolio_Value",
        ]
    )

    df = df.sort_values(
        ["Portfolio", "Date"]
    )

    if df.empty:

        raise ValueError(
            "Portfolio backtest contains no valid records."
        )

    if df.duplicated(
        subset=["Portfolio", "Date"]
    ).any():

        raise ValueError(
            "Duplicate portfolio-date records detected."
        )

    logger.info(
        "Loaded portfolio backtest: %s rows.",
        len(df),
    )

    return df


# ============================================================
# LOAD LSTM STRATEGY COMPARISON
# ============================================================

def load_strategy_comparison():

    logger.info(
        "Loading prediction-based strategy comparison."
    )

    df = pd.read_csv(
        STRATEGY_DAILY
    )

    required_columns = [
        "Date",
        "Model",
        "Net_Return",
        "Strategy_Equity",
        "Benchmark_Equity",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Strategy comparison columns missing: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    numeric_columns = [
        "Net_Return",
        "Strategy_Equity",
        "Benchmark_Equity",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Date",
            "Model",
            "Net_Return",
            "Strategy_Equity",
            "Benchmark_Equity",
        ]
    )

    df = df.sort_values(
        ["Model", "Date"]
    )

    if df.empty:

        raise ValueError(
            "Strategy comparison contains no valid records."
        )

    if df.duplicated(
        subset=["Model", "Date"]
    ).any():

        raise ValueError(
            "Duplicate strategy-date records detected."
        )

    logger.info(
        "Loaded strategy comparison: %s rows.",
        len(df),
    )

    return df


# ============================================================
# NORMALIZE STRATEGY NAMES
# ============================================================

def normalize_strategy_name(name):

    mapping = {

        "Equal Weight":
            "Equal Weight Optimized Backtest",

        "Minimum Volatility":
            "Minimum Volatility Optimized Backtest",

        "Maximum Sharpe":
            "Maximum Sharpe Optimized Backtest",

        "S&P 500":
            "S&P 500 Optimized Backtest Benchmark",

        "LSTM_Long_Only":
            "LSTM Long Only",

        "LSTM_Long_Short":
            "LSTM Long Short",

        "Equal_Weight_Buy_Hold":
            "Equal Weight Buy Hold",
    }

    return mapping.get(
        str(name),
        str(name),
    )


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    daily_returns,
    strategy_name,
    source,
):

    returns = pd.Series(
        daily_returns,
        dtype=float,
    )

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if returns.empty:

        raise ValueError(
            f"No valid returns for {strategy_name}."
        )

    if (returns <= -1).any():

        raise ValueError(
            f"Invalid return <= -100% for {strategy_name}."
        )

    observations = len(returns)

    equity = (
        INITIAL_CAPITAL
        * (1.0 + returns).cumprod()
    )

    total_return = (
        equity.iloc[-1]
        / INITIAL_CAPITAL
        - 1.0
    )

    annualized_return = (
        (1.0 + total_return)
        ** (TRADING_DAYS / observations)
        - 1.0
    )

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    if annualized_volatility > 0:

        sharpe_ratio = (
            annualized_return
            - RISK_FREE_RATE
        ) / annualized_volatility

    else:

        sharpe_ratio = np.nan

    running_max = equity.cummax()

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    maximum_drawdown = drawdown.min()

    win_rate = (
        returns > 0
    ).mean()

    return {

        "Strategy": strategy_name,

        "Source": source,

        "Observations": observations,

        "Initial_Capital": INITIAL_CAPITAL,

        "Final_Portfolio_Value": float(
            equity.iloc[-1]
        ),

        "Cumulative_Return": float(
            total_return
        ),

        "Annualized_Return": float(
            annualized_return
        ),

        "Annualized_Volatility": float(
            annualized_volatility
        ),

        "Sharpe_Ratio": float(
            sharpe_ratio
        ),

        "Maximum_Drawdown": float(
            maximum_drawdown
        ),

        "Win_Rate": float(
            win_rate
        ),
    }


# ============================================================
# PROCESS OPTIMIZED PORTFOLIOS
# ============================================================

def process_portfolios(df):

    logger.info(
        "Processing optimized portfolio results."
    )

    metrics = []

    curves = []

    for portfolio_name, group in (
        df.groupby("Portfolio")
    ):

        group = group.sort_values(
            "Date"
        ).copy()

        returns = group[
            "Daily_Return"
        ]

        strategy_name = (
            normalize_strategy_name(
                portfolio_name
            )
        )

        result = calculate_metrics(
            returns,
            strategy_name,
            "Existing Out-of-Sample Portfolio Backtest",
        )

        metrics.append(
            result
        )

        # Normalize existing portfolio equity
        # to initial capital.

        group["Normalized_Equity"] = (
            group["Portfolio_Value"]
            / INITIAL_CAPITAL
        )

        group["Integration_Strategy"] = (
            strategy_name
        )

        curves.append(
            group[
                [
                    "Date",
                    "Integration_Strategy",
                    "Normalized_Equity",
                    "Daily_Return",
                ]
            ].rename(
                columns={
                    "Integration_Strategy": "Strategy",
                    "Normalized_Equity": "Equity",
                }
            )
        )

    return metrics, curves


# ============================================================
# PROCESS LSTM STRATEGIES
# ============================================================

def process_lstm_strategies(df):

    logger.info(
        "Processing LSTM prediction-based strategies."
    )

    metrics = []

    curves = []

    selected_models = [
        "LSTM_Long_Only",
        "LSTM_Long_Short",
    ]

    for model_name in selected_models:

        group = df[
            df["Model"] == model_name
        ].copy()

        if group.empty:

            logger.warning(
                "Model not found: %s",
                model_name,
            )

            continue

        group = group.sort_values(
            "Date"
        )

        strategy_name = (
            normalize_strategy_name(
                model_name
            )
        )

        result = calculate_metrics(
            group["Net_Return"],
            strategy_name,
            "Existing LSTM Strategy Comparison",
        )

        metrics.append(
            result
        )

        group["Normalized_Equity"] = (
            group["Strategy_Equity"]
        )

        group["Integration_Strategy"] = (
            strategy_name
        )

        curves.append(
            group[
                [
                    "Date",
                    "Integration_Strategy",
                    "Normalized_Equity",
                    "Net_Return",
                ]
            ].rename(
                columns={
                    "Integration_Strategy": "Strategy",
                    "Normalized_Equity": "Equity",
                    "Net_Return": "Daily_Return",
                }
            )
        )

    # --------------------------------------------------------
    # EQUAL-WEIGHT BUY AND HOLD BENCHMARK
    # --------------------------------------------------------

    benchmark_group = df[
        df["Model"] == "Equal_Weight_Buy_Hold"
    ].copy()

    if not benchmark_group.empty:

        benchmark_group = (
            benchmark_group.sort_values("Date")
        )

        benchmark_equity = (
            benchmark_group["Benchmark_Equity"]
        )

        benchmark_returns = (
            benchmark_equity.pct_change()
        )

        # The original equity series starts after
        # its initial day's return.
        # Recover the first daily return from
        # the original daily comparison report.

        if len(benchmark_returns) > 0:

            benchmark_returns.iloc[0] = (
                benchmark_equity.iloc[0] - 1.0
            )

        benchmark_returns = (
            benchmark_returns.fillna(0.0)
        )

        benchmark_name = (
            "Equal Weight Buy Hold"
        )

        benchmark_metrics = calculate_metrics(
            benchmark_returns,
            benchmark_name,
            "Existing LSTM Strategy Comparison Benchmark",
        )

        metrics.append(
            benchmark_metrics
        )

        benchmark_group[
            "Integration_Strategy"
        ] = benchmark_name

        benchmark_group[
            "Normalized_Equity"
        ] = benchmark_equity

        curves.append(
            benchmark_group[
                [
                    "Date",
                    "Integration_Strategy",
                    "Normalized_Equity",
                ]
            ].rename(
                columns={
                    "Integration_Strategy": "Strategy",
                    "Normalized_Equity": "Equity",
                }
            )
        )

    return metrics, curves


# ============================================================
# CREATE UNIFIED COMPARISON
# ============================================================

def create_unified_comparison(
    portfolio_metrics,
    lstm_metrics,
):

    logger.info(
        "Creating unified portfolio comparison."
    )

    all_metrics = (
        portfolio_metrics
        + lstm_metrics
    )

    summary = pd.DataFrame(
        all_metrics
    )

    summary = summary.sort_values(
        "Annualized_Return",
        ascending=False,
    )

    output_path = (
        REPORT_DIR
        / "ai_portfolio_unified_comparison.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved unified comparison: %s",
        output_path,
    )

    print("\n" + "=" * 100)
    print("NORTHGATE AI PORTFOLIO UNIFIED COMPARISON")
    print("=" * 100)

    display_columns = [
        "Strategy",
        "Source",
        "Observations",
        "Cumulative_Return",
        "Annualized_Return",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Maximum_Drawdown",
        "Win_Rate",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return summary


# ============================================================
# CREATE UNIFIED DAILY EQUITY DATA
# ============================================================

def create_unified_daily(curves):

    logger.info(
        "Combining daily equity curves."
    )

    valid_curves = [
        curve
        for curve in curves
        if curve is not None
        and not curve.empty
    ]

    if not valid_curves:

        raise ValueError(
            "No daily equity curves available."
        )

    combined = pd.concat(
        valid_curves,
        ignore_index=True,
    )

    combined["Date"] = pd.to_datetime(
        combined["Date"]
    )

    combined = combined.sort_values(
        ["Strategy", "Date"]
    )

    combined = combined.dropna(
        subset=[
            "Date",
            "Strategy",
            "Equity",
        ]
    )

    output_path = (
        REPORT_DIR
        / "ai_portfolio_unified_daily.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved unified daily results: %s",
        output_path,
    )

    return combined


# ============================================================
# PLOT EQUITY CURVES
# ============================================================

def plot_unified_performance(
    combined,
):

    logger.info(
        "Generating unified portfolio performance chart."
    )

    plt.figure(
        figsize=(15, 8)
    )

    for strategy_name, group in (
        combined.groupby("Strategy")
    ):

        group = group.sort_values(
            "Date"
        )

        plt.plot(
            group["Date"],
            group["Equity"] * INITIAL_CAPITAL,
            label=strategy_name,
        )

    plt.title(
        "Northgate AI Stock Predictor - "
        "Portfolio and LSTM Strategy Comparison"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Portfolio Value (INR)"
    )

    plt.legend(
        loc="best",
        fontsize=8,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "ai_portfolio_unified_performance.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved unified performance chart: %s",
        output_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting Northgate AI portfolio integration."
    )

    validate_files()

    portfolio_df = (
        load_portfolio_backtest()
    )

    strategy_df = (
        load_strategy_comparison()
    )

    portfolio_metrics, portfolio_curves = (
        process_portfolios(
            portfolio_df
        )
    )

    lstm_metrics, lstm_curves = (
        process_lstm_strategies(
            strategy_df
        )
    )

    summary = create_unified_comparison(
        portfolio_metrics,
        lstm_metrics,
    )

    all_curves = (
        portfolio_curves
        + lstm_curves
    )

    combined_daily = (
        create_unified_daily(
            all_curves
        )
    )

    plot_unified_performance(
        combined_daily
    )

    logger.info(
        "Northgate AI portfolio integration completed."
    )

    print("\n" + "=" * 100)
    print("INTEGRATION COMPLETED")
    print("=" * 100)

    print(
        "Unified comparison:"
    )

    print(
        REPORT_DIR
        / "ai_portfolio_unified_comparison.csv"
    )

    print(
        "\nUnified daily equity:"
    )

    print(
        REPORT_DIR
        / "ai_portfolio_unified_daily.csv"
    )

    print(
        "\nPerformance chart:"
    )

    print(
        FIGURE_DIR
        / "ai_portfolio_unified_performance.png"
    )


if __name__ == "__main__":

    main()

