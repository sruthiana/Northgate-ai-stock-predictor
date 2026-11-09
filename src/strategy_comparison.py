
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NORTHGATE AI STOCK PREDICTOR
# Long-only vs Long/Short Strategy Comparison
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "deep_learning_test_predictions.csv"
)

REPORTS = ROOT / "reports"

SIGNAL_THRESHOLD = 0.0005

TRANSACTION_COST_BPS = 10

TRADING_DAYS = 252


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "Date",
        "Ticker",
        "Target_Next_Day_Log_Return",
        "Predicted_Return",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required].copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Ticker"] = df["Ticker"].astype(str)

    df["Actual_Return"] = pd.to_numeric(
        df["Target_Next_Day_Log_Return"],
        errors="coerce",
    )

    df["Predicted_Return"] = pd.to_numeric(
        df["Predicted_Return"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Ticker",
            "Actual_Return",
            "Predicted_Return",
        ]
    )

    df = df.sort_values(
        ["Ticker", "Date"]
    )

    df = df.drop_duplicates(
        ["Ticker", "Date"],
        keep="last",
    )

    print("Loaded rows:", len(df))
    print("Tickers:", sorted(df["Ticker"].unique()))
    print("Date range:", df["Date"].min(), "to", df["Date"].max())

    return df


# ============================================================
# 2. GENERATE SIGNALS
# ============================================================

def generate_positions(df):

    df = df.copy()

    prediction = df["Predicted_Return"]

    df["Signal"] = np.select(
        [
            prediction > SIGNAL_THRESHOLD,
            prediction < -SIGNAL_THRESHOLD,
        ],
        [
            "BUY",
            "SELL",
        ],
        default="HOLD",
    )

    # Long-only:
    # BUY = long
    # SELL/HOLD = cash

    df["Long_Only_Position"] = np.where(
        df["Signal"] == "BUY",
        1.0,
        0.0,
    )

    # Long-short:
    # BUY = long
    # SELL = short
    # HOLD = cash

    df["Long_Short_Position"] = np.select(
        [
            df["Signal"] == "BUY",
            df["Signal"] == "SELL",
        ],
        [
            1.0,
            -1.0,
        ],
        default=0.0,
    )

    return df


# ============================================================
# 3. CALCULATE STRATEGY RETURNS
# ============================================================

def calculate_strategy_returns(df, position_column):

    df = df.copy()

    df = df.sort_values(
        ["Ticker", "Date"]
    )

    df["Previous_Position"] = (
        df.groupby("Ticker")[position_column]
        .shift(1)
        .fillna(0.0)
    )

    # Position changes:
    # 0 -> 1 = 1 unit turnover
    # 1 -> 0 = 1 unit turnover
    # 1 -> -1 = 2 units turnover

    df["Turnover"] = (
        df[position_column]
        - df["Previous_Position"]
    ).abs()

    transaction_cost_rate = (
        TRANSACTION_COST_BPS / 10000
    )

    df["Gross_Return"] = (
        df[position_column]
        * df["Actual_Return"]
    )

    df["Transaction_Cost"] = (
        df["Turnover"]
        * transaction_cost_rate
    )

    df["Net_Return"] = (
        df["Gross_Return"]
        - df["Transaction_Cost"]
    )

    return df


# ============================================================
# 4. DAILY PORTFOLIO RETURNS
# ============================================================

def calculate_daily_returns(df, model_name):

    # Equal-weight across available stocks on each date.
    # Do not sum individual ticker returns.

    daily = (
        df.groupby("Date")
        .agg(
            Gross_Return=("Gross_Return", "mean"),
            Transaction_Cost=("Transaction_Cost", "mean"),
            Net_Return=("Net_Return", "mean"),
            Buy_Hold_Return=("Actual_Return", "mean"),
            Active_Stocks=(
                "Net_Return",
                "count",
            ),
        )
        .reset_index()
    )

    daily = daily.sort_values("Date")

    daily["Model"] = model_name

    daily["Strategy_Equity"] = (
        (1 + daily["Net_Return"]).cumprod()
    )

    daily["Benchmark_Equity"] = (
        (1 + daily["Buy_Hold_Return"]).cumprod()
    )

    daily["Strategy_Cumulative_Return"] = (
        daily["Strategy_Equity"] - 1
    )

    daily["Benchmark_Cumulative_Return"] = (
        daily["Benchmark_Equity"] - 1
    )

    daily["Strategy_Peak"] = (
        daily["Strategy_Equity"].cummax()
    )

    daily["Drawdown"] = (
        daily["Strategy_Equity"]
        / daily["Strategy_Peak"]
        - 1
    )

    return daily


# ============================================================
# 5. PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    returns,
    benchmark_returns,
    drawdown,
):

    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    benchmark_returns = pd.Series(
        benchmark_returns,
        dtype=float,
    ).dropna()

    drawdown = pd.Series(
        drawdown,
        dtype=float,
    ).dropna()

    if len(returns) == 0:

        return {}

    cumulative_return = (
        (1 + returns).prod() - 1
    )

    benchmark_return = (
        (1 + benchmark_returns).prod() - 1
    )

    annualized_return = (
        (1 + cumulative_return)
        ** (TRADING_DAYS / len(returns))
        - 1
    )

    volatility = returns.std(ddof=1)

    annualized_volatility = (
        volatility * np.sqrt(TRADING_DAYS)
        if pd.notna(volatility)
        else np.nan
    )

    sharpe = (
        returns.mean()
        / volatility
        * np.sqrt(TRADING_DAYS)
        if pd.notna(volatility)
        and volatility > 0
        else np.nan
    )

    downside_returns = returns[
        returns < 0
    ]

    downside_deviation = (
        np.sqrt(
            np.mean(
                downside_returns ** 2
            )
        )
        * np.sqrt(TRADING_DAYS)
        if len(downside_returns) > 0
        else np.nan
    )

    sortino = (
        returns.mean()
        * TRADING_DAYS
        / downside_deviation
        if pd.notna(downside_deviation)
        and downside_deviation > 0
        else np.nan
    )

    return {
        "Observations": len(returns),

        "Cumulative_Return": cumulative_return,

        "Annualized_Return": annualized_return,

        "Annualized_Volatility": annualized_volatility,

        "Sharpe_Ratio": sharpe,

        "Sortino_Ratio": sortino,

        "Maximum_Drawdown": (
            drawdown.min()
            if len(drawdown)
            else np.nan
        ),

        "Benchmark_Cumulative_Return": benchmark_return,

        "Excess_Cumulative_Return": (
            cumulative_return - benchmark_return
        ),

        "Win_Rate": (
            returns > 0
        ).mean(),

        "Average_Daily_Return": (
            returns.mean()
        ),
    }


# ============================================================
# 6. STRATEGY SUMMARY
# ============================================================

def summarize_strategy(daily):

    metrics = calculate_metrics(
        daily["Net_Return"],
        daily["Buy_Hold_Return"],
        daily["Drawdown"],
    )

    metrics["Model"] = daily["Model"].iloc[0]

    metrics["Total_Transaction_Cost"] = (
        daily["Transaction_Cost"].sum()
    )

    return metrics


# ============================================================
# 7. TICKER-LEVEL SUMMARY
# ============================================================

def summarize_tickers(df, model_name):

    results = []

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date")

        equity = (
            1 + group["Net_Return"]
        ).cumprod()

        peak = equity.cummax()

        drawdown = (
            equity / peak - 1
        )

        metrics = calculate_metrics(
            group["Net_Return"],
            group["Actual_Return"],
            drawdown,
        )

        metrics["Model"] = model_name

        metrics["Ticker"] = ticker

        metrics["Buy_Count"] = (
            group["Signal"] == "BUY"
        ).sum()

        metrics["Sell_Count"] = (
            group["Signal"] == "SELL"
        ).sum()

        metrics["Hold_Count"] = (
            group["Signal"] == "HOLD"
        ).sum()

        results.append(metrics)

    return pd.DataFrame(results)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("NORTHGATE AI STOCK PREDICTOR")
    print("LONG-ONLY VS LONG-SHORT COMPARISON")
    print("=" * 65)

    df = load_data()

    df = generate_positions(df)

    all_daily = []
    all_ticker_results = []
    all_summary = []

    strategies = [
        (
            "LSTM_Long_Only",
            "Long_Only_Position",
        ),
        (
            "LSTM_Long_Short",
            "Long_Short_Position",
        ),
    ]

    for model_name, position_column in strategies:

        print("\nEvaluating:", model_name)

        strategy = calculate_strategy_returns(
            df,
            position_column,
        )

        daily = calculate_daily_returns(
            strategy,
            model_name,
        )

        summary = summarize_strategy(
            daily
        )

        ticker_summary = summarize_tickers(
            strategy,
            model_name,
        )

        all_daily.append(daily)

        all_summary.append(summary)

        all_ticker_results.append(
            ticker_summary
        )

        print("\nPortfolio metrics:")

        print(
            pd.DataFrame([summary]).to_string(
                index=False
            )
        )

    # ========================================================
    # EQUAL-WEIGHT BUY-AND-HOLD BENCHMARK
    # ========================================================

    benchmark = (
        df.groupby("Date")["Actual_Return"]
        .mean()
        .reset_index()
    )

    benchmark = benchmark.sort_values("Date")

    benchmark["Benchmark_Equity"] = (
        (1 + benchmark["Actual_Return"]).cumprod()
    )

    benchmark["Benchmark_Cumulative_Return"] = (
        benchmark["Benchmark_Equity"] - 1
    )

    benchmark["Model"] = "Equal_Weight_Buy_Hold"

    benchmark["Net_Return"] = (
        benchmark["Actual_Return"]
    )

    benchmark["Buy_Hold_Return"] = (
        benchmark["Actual_Return"]
    )

    benchmark["Drawdown"] = (
        benchmark["Benchmark_Equity"]
        / benchmark["Benchmark_Equity"].cummax()
        - 1
    )

    benchmark_summary = calculate_metrics(
        benchmark["Actual_Return"],
        benchmark["Actual_Return"],
        benchmark["Drawdown"],
    )

    benchmark_summary["Model"] = (
        "Equal_Weight_Buy_Hold"
    )

    benchmark_summary["Total_Transaction_Cost"] = 0.0

    all_daily.append(benchmark)

    all_summary.append(
        benchmark_summary
    )

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    daily_results = pd.concat(
        all_daily,
        ignore_index=True,
    )

    summary_results = pd.DataFrame(
        all_summary
    )

    ticker_results = pd.concat(
        all_ticker_results,
        ignore_index=True,
    )

    daily_results.to_csv(
        REPORTS / "strategy_comparison_daily.csv",
        index=False,
    )

    summary_results.to_csv(
        REPORTS / "strategy_comparison_summary.csv",
        index=False,
    )

    ticker_results.to_csv(
        REPORTS / "strategy_comparison_by_ticker.csv",
        index=False,
    )

    # Save individual ticker-level strategy returns
    df.to_csv(
        REPORTS / "strategy_comparison_signals.csv",
        index=False,
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 65)

    print(
        summary_results[
            [
                "Model",
                "Cumulative_Return",
                "Annualized_Return",
                "Annualized_Volatility",
                "Sharpe_Ratio",
                "Maximum_Drawdown",
                "Benchmark_Cumulative_Return",
                "Excess_Cumulative_Return",
                "Total_Transaction_Cost",
            ]
        ].to_string(index=False)
    )

    print("\nTICKER-LEVEL PERFORMANCE")

    print(
        ticker_results[
            [
                "Model",
                "Ticker",
                "Cumulative_Return",
                "Sharpe_Ratio",
                "Maximum_Drawdown",
                "Buy_Count",
                "Sell_Count",
                "Hold_Count",
            ]
        ].to_string(index=False)
    )

    print("\nSaved reports:")

    print(
        REPORTS / "strategy_comparison_daily.csv"
    )

    print(
        REPORTS / "strategy_comparison_summary.csv"
    )

    print(
        REPORTS / "strategy_comparison_by_ticker.csv"
    )

    print(
        REPORTS / "strategy_comparison_signals.csv"
    )

    print("\nSTRATEGY COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
