from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

VALIDATION_FILE = REPORTS / "direction_validation_by_ticker.csv"
TEST_FILE = REPORTS / "direction_test_predictions.csv"
OUTPUT_FILE = REPORTS / "direction_shortlist_test_results.csv"

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
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    # Select candidates using validation results ONLY.
    selected = validation.loc[
        validation["Macro_F1_Improvement"] > 0
    ].sort_values(
        "Macro_F1_Improvement", ascending=False
    ).head(4)["Ticker"].tolist()

    print("Selected using validation only:", selected)

    rows = []
    for ticker in selected:
        group = test[test["Ticker"] == ticker].copy()
        actual = group["Actual_Direction"]
        predicted = group["Predicted_Direction"]

        majority_class = actual.mode().iloc[0]
        baseline_pred = [majority_class] * len(group)

        rows.append({
            "Ticker": ticker,
            "Observations": len(group),
            "Test_Majority_Class": majority_class,
            "Test_Baseline_Accuracy": accuracy_score(
                actual, baseline_pred
            ),
            "Test_Baseline_Macro_F1": macro_f1(
                actual, baseline_pred
            ),
            "Test_Model_Accuracy": accuracy_score(
                actual, predicted
            ),
            "Test_Model_Balanced_Accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "Test_Model_Macro_F1": macro_f1(
                actual, predicted
            ),
            "Test_Macro_F1_Improvement": (
                macro_f1(actual, predicted)
                - macro_f1(actual, baseline_pred)
            ),
            "Test_Down_Recall": (
                ((actual == "Down") & (predicted == "Down")).sum()
                / max((actual == "Down").sum(), 1)
            ),
            "Test_Neutral_Recall": (
                ((actual == "Neutral") & (predicted == "Neutral")).sum()
                / max((actual == "Neutral").sum(), 1)
            ),
            "Test_Up_Recall": (
                ((actual == "Up") & (predicted == "Up")).sum()
                / max((actual == "Up").sum(), 1)
            ),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSHORTLIST TEST RESULTS")
    print(result.to_string(index=False))

    print("\nSHORTLIST TEST PREDICTION DISTRIBUTION")
    print(
        test[test["Ticker"].isin(selected)]
        .groupby(["Ticker", "Predicted_Direction"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nSAVED:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
