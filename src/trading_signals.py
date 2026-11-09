
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NORTHGATE AI STOCK PREDICTOR
# Trading signals and historical strategy evaluation
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "reports"
PREDICTIONS = ROOT / "data" / "predictions"

LSTM_FILE = REPORTS / "deep_learning_test_predictions.csv"
RIDGE_FILE = PREDICTIONS / "ridge_test_predictions.parquet"

OUTPUT_SIGNALS = REPORTS / "trading_signals.csv"
OUTPUT_DAILY = REPORTS / "trading_strategy_daily.csv"
OUTPUT_SUMMARY = REPORTS / "trading_strategy_summary.csv"
OUTPUT_TICKER = REPORTS / "trading_strategy_by_ticker.csv"

# Initial experimental trading assumptions
SIGNAL_THRESHOLD = 0.0005
TRANSACTION_COST_BPS = 10
ANNUALIZATION_FACTOR = 252


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

def load_predictions():
    if not LSTM_FILE.exists():
        raise FileNotFoundError(
            f"LSTM predictions not found: {LSTM_FILE}"
        )

    if not RIDGE_FILE.exists():
        raise FileNotFoundError(
            f"Ridge predictions not found: {RIDGE_FILE}"
        )

    lstm = pd.read_csv(LSTM_FILE)

    ridge = pd.read_parquet(RIDGE_FILE)

    lstm = lstm[
        [
            "Date",
            "Ticker",
            "Target_Next_Day_Log_Return",
            "Predicted_Return",
        ]
    ].copy()

    ridge = ridge[
        [
            "Date",
            "Ticker",
            "Target_Next_Day_Log_Return",
            "Predicted_Return",
        ]
    ].copy()

    lstm["Model"] = "LSTM"
    ridge["Model"] = "Ridge"

    data = pd.concat(
        [lstm, ridge],
        ignore_index=True,
    )

    data["Date"] = pd.to_datetime(data["Date"])

    data["Ticker"] = data["Ticker"].astype(str)

    data["Actual_Return"] = pd.to_numeric(
        data["Target_Next_Day_Log_Return"],
        errors="coerce",
    )

    data["Predicted_Return"] = pd.to_numeric(
        data["Predicted_Return"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "Date",
            "Ticker",
            "Actual_Return",
            "Predicted_Return",
        ]
    )

    data = data.sort_values(
        ["Model", "Ticker", "Date"]
    )

    data = data.drop_duplicates(
        ["Model", "Ticker", "Date"],
        keep="last",
    )

    return data


# ============================================================
# 2. GENERATE BUY / HOLD / SELL SIGNALS
# ============================================================

def generate_signals(data):
    data = data.copy()

    data["Signal"] = np.select(
        [
            data["Predicted_Return"] > SIGNAL_THRESHOLD,
            data["Predicted_Return"] < -SIGNAL_THRESHOLD,
        ],
        [
            "BUY",
            "SELL",
        ],
        default="HOLD",
    )

    data["Position"] = data["Signal"].map(
        {
            "BUY": 1.0,
            "HOLD": 0.0,
            "SELL": -1.0,
        }
    )

    data["Actual_Direction"] = np.sign(
        data["Actual_Return"]
    )

    data["Predicted_Direction"] = np.sign(
        data["Predicted_Return"]
    )

    data["Direction_Correct"] = (
        data["Actual_Direction"]
        == data["Predicted_Direction"]
    )

    data["Signal_Correct"] = (
        data["Position"]
        == data["Actual_Direction"]
    )

    return data


# ============================================================
# 3. CALCULATE PER-TICKER STRATEGY RETURNS
# ============================================================

def calculate_ticker_returns(data):
    data = data.copy()

    data = data.sort_values(
        ["Model", "Ticker", "Date"]
    )

    data["Previous_Position"] = (
        data.groupby(
            ["Model", "Ticker"]
        )["Position"]
        .shift(1)
        .fillna(0.0)
    )

    data["Turnover"] = (
        data["Position"]
        - data["Previous_Position"]
    ).abs()

    transaction_cost = (
        TRANSACTION_COST_BPS / 10000
    )

    data["Gross_Strategy_Return"] = (
        data["Position"]
        * data["Actual_Return"]
    )

    data["Transaction_Cost"] = (
        data["Turnover"]
        * transaction_cost
    )

    data["Net_Strategy_Return"] = (
        data["Gross_Strategy_Return"]
        - data["Transaction_Cost"]
    )

    data["Buy_Hold_Return"] = (
        data["Actual_Return"]
    )

    return data


# ============================================================
# 4. PORTFOLIO-LEVEL DAILY RETURNS
# ============================================================

def calculate_daily_portfolio(data):
    daily = (
        data.groupby(
            ["Model", "Date"],
            as_index=False,
        )
        .agg(
            Portfolio_Gross_Return=(
                "Gross_Strategy_Return",
                "mean",
            ),
            Portfolio_Transaction_Cost=(
                "Transaction_Cost",
                "mean",
            ),
            Portfolio_Net_Return=(
                "Net_Strategy_Return",
                "mean",
            ),
            Buy_Hold_Return=(
                "Buy_Hold_Return",
                "mean",
            ),
            Average_Position=(
                "Position",
                "mean",
            ),
            Active_Assets=(
                "Position",
                lambda x: int((x != 0).sum()),
            ),
            Signal_Count=(
                "Signal",
                "count",
            ),
        )
    )

    daily = daily.sort_values(
        ["Model", "Date"]
    )

    daily["Strategy_Cumulative_Return"] = (
        daily.groupby("Model")[
            "Portfolio_Net_Return"
        ]
        .transform(
            lambda x: (
                (1 + x).cumprod() - 1
            )
        )
    )

    daily["Buy_Hold_Cumulative_Return"] = (
        daily.groupby("Model")[
            "Buy_Hold_Return"
        ]
        .transform(
            lambda x: (
                (1 + x).cumprod() - 1
            )
        )
    )

    daily["Strategy_Equity"] = (
        1 + daily["Strategy_Cumulative_Return"]
    )

    daily["Strategy_Peak"] = (
        daily.groupby("Model")[
            "Strategy_Equity"
        ]
        .cummax()
    )

    daily["Strategy_Drawdown"] = (
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

    if len(returns) == 0:
        return {}

    cumulative_return = (
        (1 + returns).prod() - 1
    )

    benchmark_cumulative_return = (
        (1 + benchmark_returns).prod() - 1
    )

    annualized_return = (
        (1 + cumulative_return)
        ** (
            ANNUALIZATION_FACTOR
            / len(returns)
        )
        - 1
    )

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
        if len(returns) > 1
        else np.nan
    )

    sharpe_ratio = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
        if len(returns) > 1
        and returns.std(ddof=1) > 0
        else np.nan
    )

    downside = returns[
        returns < 0
    ]

    downside_deviation = (
        np.sqrt(
            np.mean(
                downside ** 2
            )
        )
        * np.sqrt(ANNUALIZATION_FACTOR)
        if len(downside) > 0
        else np.nan
    )

    sortino_ratio = (
        returns.mean()
        * ANNUALIZATION_FACTOR
        / downside_deviation
        if downside_deviation > 0
        else np.nan
    )

    return {
        "Observations": len(returns),

        "Cumulative_Return": cumulative_return,

        "Annualized_Return": annualized_return,

        "Annualized_Volatility": annualized_volatility,

        "Sharpe_Ratio": sharpe_ratio,

        "Sortino_Ratio": sortino_ratio,

        "Maximum_Drawdown": drawdown.min(),

        "Benchmark_Cumulative_Return": (
            benchmark_cumulative_return
        ),

        "Excess_Cumulative_Return": (
            cumulative_return
            - benchmark_cumulative_return
        ),

        "Win_Rate": (
            (returns > 0).mean()
        ),

        "Average_Daily_Return": (
            returns.mean()
        ),
    }


# ============================================================
# 6. BUILD PORTFOLIO SUMMARY
# ============================================================

def build_summary(daily):
    results = []

    for model, group in daily.groupby("Model"):
        group = group.sort_values("Date")

        metrics = calculate_metrics(
            group["Portfolio_Net_Return"],
            group["Buy_Hold_Return"],
            group["Strategy_Drawdown"],
        )

        metrics["Model"] = model

        metrics["Total_Transaction_Cost"] = (
            group["Portfolio_Transaction_Cost"].sum()
        )

        metrics["Average_Active_Assets"] = (
            group["Active_Assets"].mean()
        )

        results.append(metrics)

    summary = pd.DataFrame(results)

    if not summary.empty:
        summary = summary.sort_values(
            "Sharpe_Ratio",
            ascending=False,
        )

    return summary


# ============================================================
# 7. BUILD PER-TICKER SUMMARY
# ============================================================

def build_ticker_summary(data):
    results = []

    for (model, ticker), group in data.groupby(
        ["Model", "Ticker"]
    ):
        group = group.sort_values("Date")

        equity = (
            1 + group["Net_Strategy_Return"]
        ).cumprod()

        peak = equity.cummax()

        drawdown = (
            equity / peak - 1
        )

        metrics = calculate_metrics(
            group["Net_Strategy_Return"],
            group["Buy_Hold_Return"],
            drawdown,
        )

        metrics["Model"] = model
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

        metrics["Total_Transaction_Cost"] = (
            group["Transaction_Cost"].sum()
        )

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

    PREDICTIONS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("NORTHGATE AI STOCK PREDICTOR")
    print("TRADING SIGNAL GENERATION")
    print("=" * 65)

    print("\nLoading predictions...")

    predictions = load_predictions()

    print(
        "Loaded prediction rows:",
        len(predictions),
    )

    print(
        "Models:",
        predictions["Model"].unique().tolist(),
    )

    print("\nGenerating trading signals...")

    signals = generate_signals(
        predictions
    )

    signals = calculate_ticker_returns(
        signals
    )

    signals.to_csv(
        OUTPUT_SIGNALS,
        index=False,
    )

    print(
        "Saved signals:",
        OUTPUT_SIGNALS,
    )

    print("\nCalculating portfolio returns...")

    daily = calculate_daily_portfolio(
        signals
    )

    daily.to_csv(
        OUTPUT_DAILY,
        index=False,
    )

    print(
        "Saved daily portfolio returns:",
        OUTPUT_DAILY,
    )

    print("\nCalculating portfolio metrics...")

    summary = build_summary(
        daily
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        "Saved portfolio summary:",
        OUTPUT_SUMMARY,
    )

    ticker_summary = build_ticker_summary(
        signals
    )

    ticker_summary.to_csv(
        OUTPUT_TICKER,
        index=False,
    )

    print(
        "Saved ticker summary:",
        OUTPUT_TICKER,
    )

    print("\nPORTFOLIO PERFORMANCE")

    if not summary.empty:
        print(
            summary.to_string(
                index=False
            )
        )

    print("\nPER-TICKER PERFORMANCE")

    if not ticker_summary.empty:
        print(
            ticker_summary[
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
            ].to_string(
                index=False
            )
        )

    print("\nSIGNAL DISTRIBUTION")

    print(
        signals.groupby(
            ["Model", "Signal"]
        ).size().to_string()
    )

    print("\n" + "=" * 65)
    print("TRADING SIGNAL GENERATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
