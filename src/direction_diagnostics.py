
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUT = REPORTS / "direction_test_predictions.csv"
OUTPUT = REPORTS / "direction_test_by_ticker.csv"

LABELS = ["Down", "Neutral", "Up"]


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

        results.append({
            "Ticker": ticker,
            "Observations": len(group),
            "Accuracy": accuracy_score(actual, predicted),
            "Balanced_Accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "Macro_F1": f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "Down_Recall": (
                (actual == "Down") & (predicted == "Down")
            ).sum() / max((actual == "Down").sum(), 1),
            "Neutral_Recall": (
                (actual == "Neutral") & (predicted == "Neutral")
            ).sum() / max((actual == "Neutral").sum(), 1),
            "Up_Recall": (
                (actual == "Up") & (predicted == "Up")
            ).sum() / max((actual == "Up").sum(), 1),
        })

    result = pd.DataFrame(results)
    result = result.sort_values("Macro_F1", ascending=False)

    result.to_csv(OUTPUT, index=False)

    print("\nTEST PERFORMANCE BY TICKER")
    print(result.to_string(index=False))

    print("\nPREDICTED CLASS DISTRIBUTION")
    print(
        df.groupby(["Ticker", "Predicted_Direction"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nOVERALL CLASSIFICATION REPORT")
    print(
        classification_report(
            df["Actual_Direction"],
            df["Predicted_Direction"],
            labels=LABELS,
            zero_division=0,
        )
    )

    print("\nSAVED:", OUTPUT)


if __name__ == "__main__":
    main()
