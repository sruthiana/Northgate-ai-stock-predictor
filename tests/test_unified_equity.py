import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DAILY_FILE = ROOT / "reports" / "ai_portfolio_unified_daily.csv"

EXPECTED_STRATEGIES = {
    "Equal Weight Optimized Backtest",
    "LSTM Long Only",
    "LSTM Long Short",
    "Maximum Sharpe Optimized Backtest",
    "Minimum Volatility Optimized Backtest",
    "S&P 500 Optimized Backtest Benchmark",
}


class TestUnifiedPortfolioEquity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not DAILY_FILE.exists():
            raise FileNotFoundError(
                f"Unified daily results not found: {DAILY_FILE}"
            )

        cls.data = pd.read_csv(DAILY_FILE)

    def test_required_columns(self):
        required = {"Date", "Strategy", "Equity", "Daily_Return"}
        self.assertTrue(required.issubset(self.data.columns))

    def test_expected_strategies_present(self):
        actual = set(self.data["Strategy"].unique())
        self.assertEqual(EXPECTED_STRATEGIES, actual)

    def test_no_missing_values(self):
        self.assertEqual(int(self.data.isna().sum().sum()), 0)

    def test_equity_values_positive(self):
        self.assertTrue((self.data["Equity"] > 0).all())

    def test_unique_dates_per_strategy(self):
        duplicates = self.data.duplicated(
            subset=["Strategy", "Date"]
        )
        self.assertFalse(duplicates.any())

    def test_equity_matches_compounded_returns(self):
        for strategy, group in self.data.groupby("Strategy"):
            group = group.sort_values("Date")

            compounded_return = (
                (1 + group["Daily_Return"]).prod() - 1
            )

            equity_return = group["Equity"].iloc[-1] - 1

            self.assertAlmostEqual(
                compounded_return,
                equity_return,
                places=8,
                msg=f"Equity reconciliation failed for {strategy}",
            )


if __name__ == "__main__":
    unittest.main()
