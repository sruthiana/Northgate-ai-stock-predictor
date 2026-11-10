
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUT = REPORTS / "direction_validation_predictions.csv"
OUTPUT = REPORTS / "direction_validation_by_ticker.csv"

LABELS = ["Down", "Neutral", "Up"]


def macro_f1(actual, predicted):
    return f1_score(
        actual,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


def main():
    df = pd.read_csv(INPUT)

    required = [
        "Ticker",
        "Actual_Direction",
        "Predicted_Direction",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for ticker, group in df.groupby("Ticker"):
        actual = group["Actual_Direction"]
        predicted = group["Predicted_Direction"]

        # Majority class baseline for this ticker
        majority = actual.mode().iloc[0]
        baseline_predictions = [majority] * len(actual)

        results.append({
            "Ticker": ticker,
            "Observations": len(group),
            "Actual_Majority_Class": majority,
            "Baseline_Accuracy": accuracy_score(
                actual, baseline_predictions
            ),
            "Baseline_Macro_F1": macro_f1(
                actual, baseline_predictions
            ),
            "Model_Accuracy": accuracy_score(
                actual, predicted
            ),
            "Model_Balanced_Accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "Model_Macro_F1": macro_f1(
                actual, predicted
            ),
            "Model_Down_Recall": (
                ((actual == "Down") & (predicted == "Down")).sum()
                / max((actual == "Down").sum(), 1)
            ),
            "Model_Neutral_Recall": (
                ((actual == "Neutral") & (predicted == "Neutral")).sum()
                / max((actual == "Neutral").sum(), 1)
            ),
            "Model_Up_Recall": (
                ((actual == "Up") & (predicted == "Up")).sum()
                / max((actual == "Up").sum(), 1)
            ),
        })

    result = pd.DataFrame(results)

    result["Macro_F1_Improvement"] = (
        result["Model_Macro_F1"]
        - result["Baseline_Macro_F1"]
    )

    result = result.sort_values(
        "Macro_F1_Improvement",
        ascending=False,
    )

    result.to_csv(OUTPUT, index=False)

    print("\nVALIDATION PERFORMANCE BY TICKER")
    print(result.to_string(index=False))

    print("\nTICKERS BEATING THEIR MAJORITY BASELINE")
    better = result[result["Macro_F1_Improvement"] > 0]

    if better.empty:
        print("None")
    else:
        print(
            better[
                [
                    "Ticker",
                    "Model_Macro_F1",
                    "Baseline_Macro_F1",
                    "Macro_F1_Improvement",
                ]
            ].to_string(index=False)
        )

    print("\nSAVED:", OUTPUT)


if __name__ == "__main__":
    main()
