import pandas as pd
import numpy as np

path = "reports/ai_portfolio_unified_daily.csv"
df = pd.read_csv(path, parse_dates=["Date"])

print("=" * 70)
print("FINAL EQUITY CURVE VALIDATION")
print("=" * 70)

for strategy, group in df.groupby("Strategy"):
    group = group.sort_values("Date").reset_index(drop=True)

    equity = group["Equity"]
    returns = group["Daily_Return"]

    compounded_return = (1 + returns).prod() - 1
    equity_return = equity.iloc[-1] / equity.iloc[0] - 1

    duplicate_dates = group["Date"].duplicated().sum()
    nonpositive_equity = (equity <= 0).sum()

    print(f"\nStrategy: {strategy}")
    print(f"Observations: {len(group)}")
    print(f"Duplicate dates: {duplicate_dates}")
    print(f"Non-positive equity values: {nonpositive_equity}")
    print(f"Compounded daily return: {compounded_return:.6f}")
    print(f"Equity-based return: {equity_return:.6f}")
    print(f"Difference: {abs(compounded_return-equity_return):.10f}")

    if duplicate_dates == 0 and nonpositive_equity == 0:
        print("Structural checks: PASS")
    else:
        print("Structural checks: REVIEW")

print("\nValidation completed.")
