"""
Northgate AI Stock Predictor
Feature Signal Analysis

Purpose:
1. Analyze feature-target correlations.
2. Identify potential predictive features.
3. Compare feature signals across stocks.
4. Summarize feature distributions.
5. Use training and validation data only.

No test data is used.
No models are trained.
Existing files are not modified.
"""

import logging

import numpy as np
import pandas as pd

from src.config import project_path


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


TRAIN_PATH = project_path(
    "data/processed/train.parquet"
)

VALIDATION_PATH = project_path(
    "data/processed/validation.parquet"
)

CORRELATION_OUTPUT = project_path(
    "reports/feature_target_correlations.csv"
)

TICKER_SIGNAL_OUTPUT = project_path(
    "reports/feature_signal_by_ticker.csv"
)

DISTRIBUTION_OUTPUT = project_path(
    "reports/feature_distribution_summary.csv"
)


# --------------------------------------------------
# 2. TARGET CONFIGURATION
# --------------------------------------------------

TARGET_CANDIDATES = [
    "Target_Next_Day_Log_Return",
    "target_next_day_log_return",
    "next_day_log_return",
]


def identify_target(df):

    for column in TARGET_CANDIDATES:

        if column in df.columns:
            return column

    raise ValueError(
        "Target column not found. "
        f"Available columns: {list(df.columns)}"
    )


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

def load_dataset(path, dataset_name):

    logger.info(
        "Loading %s dataset.",
        dataset_name,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    if "Date" not in df.columns:

        raise ValueError(
            f"{dataset_name} dataset has no Date column."
        )

    if "Ticker" not in df.columns:

        raise ValueError(
            f"{dataset_name} dataset has no Ticker column."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date", "Ticker"]
    )

    target = identify_target(df)

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    df = df.dropna(
        subset=[target]
    )

    if df.empty:

        raise ValueError(
            f"{dataset_name} dataset is empty."
        )

    logger.info(
        "%s dataset loaded: %s rows, %s columns.",
        dataset_name,
        len(df),
        len(df.columns),
    )

    return df, target


# --------------------------------------------------
# 4. IDENTIFY NUMERIC FEATURES
# --------------------------------------------------

def identify_features(df, target):

    excluded_columns = {
        "Date",
        "Ticker",
        target,
        "Patient_ID",
    }

    numeric_features = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        column
        for column in numeric_features
        if column not in excluded_columns
    ]

    if not features:

        raise ValueError(
            "No numeric features found."
        )

    logger.info(
        "Identified %s numeric features.",
        len(features),
    )

    return features


# --------------------------------------------------
# 5. FEATURE-TARGET CORRELATIONS
# --------------------------------------------------

def calculate_correlations(
    train_df,
    validation_df,
    train_target,
    validation_target,
    features,
):

    logger.info(
        "Calculating feature-target correlations."
    )

    results = []

    datasets = [
        (
            "train",
            train_df,
            train_target,
        ),
        (
            "validation",
            validation_df,
            validation_target,
        ),
    ]

    for dataset_name, df, target in datasets:

        for feature in features:

            feature_values = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            target_values = pd.to_numeric(
                df[target],
                errors="coerce",
            )

            valid = (
                feature_values.notna()
                & target_values.notna()
                & np.isfinite(feature_values)
                & np.isfinite(target_values)
            )

            x = feature_values[valid]

            y = target_values[valid]

            observations = len(x)

            if observations < 2:

                correlation = np.nan

            elif x.nunique() < 2:

                correlation = np.nan

            elif y.nunique() < 2:

                correlation = np.nan

            else:

                correlation = x.corr(y)

            results.append({

                "Dataset": dataset_name,

                "Feature": feature,

                "Observations": observations,

                "Feature_Mean": (
                    x.mean()
                    if observations
                    else np.nan
                ),

                "Feature_Std": (
                    x.std()
                    if observations > 1
                    else np.nan
                ),

                "Target_Mean": (
                    y.mean()
                    if observations
                    else np.nan
                ),

                "Pearson_Correlation": correlation,

                "Absolute_Correlation": (
                    abs(correlation)
                    if pd.notna(correlation)
                    else np.nan
                ),

            })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        [
            "Dataset",
            "Absolute_Correlation",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )

    result_df.to_csv(
        CORRELATION_OUTPUT,
        index=False,
    )

    logger.info(
        "Saved feature-target correlations: %s",
        CORRELATION_OUTPUT,
    )

    return result_df


# --------------------------------------------------
# 6. FEATURE SIGNAL BY TICKER
# --------------------------------------------------

def calculate_ticker_signals(
    train_df,
    target,
    features,
):

    logger.info(
        "Calculating feature signals by ticker."
    )

    results = []

    for ticker, group in train_df.groupby(
        "Ticker"
    ):

        for feature in features:

            x = pd.to_numeric(
                group[feature],
                errors="coerce",
            )

            y = pd.to_numeric(
                group[target],
                errors="coerce",
            )

            valid = (
                x.notna()
                & y.notna()
                & np.isfinite(x)
                & np.isfinite(y)
            )

            x = x[valid]

            y = y[valid]

            observations = len(x)

            if observations < 2:

                correlation = np.nan

            elif x.nunique() < 2:

                correlation = np.nan

            elif y.nunique() < 2:

                correlation = np.nan

            else:

                correlation = x.corr(y)

            results.append({

                "Ticker": ticker,

                "Feature": feature,

                "Observations": observations,

                "Feature_Mean": (
                    x.mean()
                    if observations
                    else np.nan
                ),

                "Feature_Std": (
                    x.std()
                    if observations > 1
                    else np.nan
                ),

                "Target_Mean": (
                    y.mean()
                    if observations
                    else np.nan
                ),

                "Pearson_Correlation": correlation,

                "Absolute_Correlation": (
                    abs(correlation)
                    if pd.notna(correlation)
                    else np.nan
                ),

            })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        [
            "Ticker",
            "Absolute_Correlation",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )

    result_df.to_csv(
        TICKER_SIGNAL_OUTPUT,
        index=False,
    )

    logger.info(
        "Saved ticker feature signals: %s",
        TICKER_SIGNAL_OUTPUT,
    )

    return result_df


# --------------------------------------------------
# 7. FEATURE DISTRIBUTION SUMMARY
# --------------------------------------------------

def calculate_feature_distributions(
    train_df,
    validation_df,
    features,
):

    logger.info(
        "Calculating feature distribution summaries."
    )

    results = []

    datasets = [
        (
            "train",
            train_df,
        ),
        (
            "validation",
            validation_df,
        ),
    ]

    for dataset_name, df in datasets:

        for feature in features:

            values = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            values = values[
                values.notna()
                & np.isfinite(values)
            ]

            if values.empty:

                results.append({

                    "Dataset": dataset_name,

                    "Feature": feature,

                    "Observations": 0,

                    "Missing_Count": len(df),

                    "Mean": np.nan,

                    "Std": np.nan,

                    "Min": np.nan,

                    "Q25": np.nan,

                    "Median": np.nan,

                    "Q75": np.nan,

                    "Max": np.nan,

                    "Unique_Values": 0,

                })

                continue

            results.append({

                "Dataset": dataset_name,

                "Feature": feature,

                "Observations": len(values),

                "Missing_Count": (
                    len(df) - len(values)
                ),

                "Mean": values.mean(),

                "Std": values.std(),

                "Min": values.min(),

                "Q25": values.quantile(0.25),

                "Median": values.median(),

                "Q75": values.quantile(0.75),

                "Max": values.max(),

                "Unique_Values": values.nunique(),

            })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        DISTRIBUTION_OUTPUT,
        index=False,
    )

    logger.info(
        "Saved feature distribution summary: %s",
        DISTRIBUTION_OUTPUT,
    )

    return result_df


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():

    logger.info(
        "Starting feature signal analysis."
    )

    train_df, train_target = load_dataset(
        TRAIN_PATH,
        "training",
    )

    validation_df, validation_target = load_dataset(
        VALIDATION_PATH,
        "validation",
    )

    if train_target != validation_target:

        raise ValueError(
            "Training and validation target columns "
            "do not match."
        )

    target = train_target

    features = identify_features(
        train_df,
        target,
    )

    # Ensure validation contains the same features.

    missing_validation_features = [
        feature
        for feature in features
        if feature not in validation_df.columns
    ]

    if missing_validation_features:

        raise ValueError(
            "Validation dataset is missing features: "
            f"{missing_validation_features}"
        )

    # --------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------

    correlations = calculate_correlations(
        train_df,
        validation_df,
        target,
        validation_target,
        features,
    )

    # --------------------------------------------------
    # TICKER SIGNALS
    # --------------------------------------------------

    ticker_signals = calculate_ticker_signals(
        train_df,
        target,
        features,
    )

    # --------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------

    distributions = calculate_feature_distributions(
        train_df,
        validation_df,
        features,
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 75)

    print("TOP FEATURE-TARGET CORRELATIONS")

    print("=" * 75)

    for dataset_name in [
        "train",
        "validation",
    ]:

        subset = correlations[
            correlations["Dataset"]
            == dataset_name
        ].head(15)

        print(
            f"\n{dataset_name.upper()} DATASET"
        )

        print(
            subset[
                [
                    "Feature",
                    "Observations",
                    "Pearson_Correlation",
                    "Absolute_Correlation",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print("\n" + "=" * 75)

    print("TOP FEATURE SIGNALS BY TICKER")

    print("=" * 75)

    for ticker in sorted(
        ticker_signals["Ticker"].unique()
    ):

        subset = ticker_signals[
            ticker_signals["Ticker"]
            == ticker
        ].head(5)

        print(
            f"\n{ticker}"
        )

        print(
            subset[
                [
                    "Feature",
                    "Pearson_Correlation",
                    "Absolute_Correlation",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print("\n" + "=" * 75)

    print("FEATURE DISTRIBUTION SUMMARY")

    print("=" * 75)

    print(
        distributions[
            distributions["Dataset"]
            == "train"
        ][
            [
                "Feature",
                "Missing_Count",
                "Mean",
                "Std",
                "Unique_Values",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    logger.info(
        "Feature signal analysis completed."
    )


if __name__ == "__main__":

    main()
