
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    GRU,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
PREDICTION_DIR = ROOT_DIR / "data" / "predictions"

for directory in [
    MODEL_DIR,
    REPORT_DIR,
    PREDICTION_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


SEED = 42
SEQUENCE_LENGTH = 60
EPOCHS = 40
BATCH_SIZE = 128

TARGET = "Target_Next_Day_Log_Return"

FEATURES = [
    "Daily_Return",
    "Log_Return",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Std",
    "Volatility_20",
    "Volatility_60",
    "DGS10",
    "DGS3MO",
    "Treasury_Spread_10Y_3M",
    "CPIAUCSL",
    "CPI_YoY",
    "UNRATE",
    "UNRATE_Change",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(name):

    path = DATA_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    required = [
        "Date",
        "Ticker",
        TARGET,
    ] + FEATURES

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    logger.info(
        "%s dataset shape: %s",
        name,
        df.shape,
    )

    return df


# ============================================================
# LOAD ALL SPLITS
# ============================================================

def load_all_data():

    train = load_dataset("train")
    validation = load_dataset("validation")
    test = load_dataset("test")

    train_max = train["Date"].max()
    validation_min = validation["Date"].min()

    validation_max = validation["Date"].max()
    test_min = test["Date"].min()

    if train_max >= validation_min:
        raise ValueError(
            "Training and validation dates overlap."
        )

    if validation_max >= test_min:
        raise ValueError(
            "Validation and test dates overlap."
        )

    logger.info(
        "Chronological split verified."
    )

    return train, validation, test


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_data(train, validation, test):

    logger.info(
        "Preparing numerical features."
    )

    scaler = StandardScaler()

    train_features = train[FEATURES].copy()

    validation_features = validation[FEATURES].copy()

    test_features = test[FEATURES].copy()

    for frame in [
        train_features,
        validation_features,
        test_features,
    ]:

        frame.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

    # Learn imputation values from training data only.
    medians = train_features.median()

    train_features = train_features.fillna(
        medians
    )

    validation_features = validation_features.fillna(
        medians
    )

    test_features = test_features.fillna(
        medians
    )

    if train_features.isna().any().any():

        raise ValueError(
            "Training features still contain missing values."
        )

    if train_features.empty:

        raise ValueError(
            "Training dataset is empty."
        )

    # Fit scaler only on training observations.
    scaler.fit(train_features)

    train_scaled = scaler.transform(
        train_features
    )

    validation_scaled = scaler.transform(
        validation_features
    )

    test_scaled = scaler.transform(
        test_features
    )

    # Fixed ticker encoding based on training universe.
    tickers = sorted(
        train["Ticker"].unique()
    )

    ticker_to_index = {
        ticker: index
        for index, ticker in enumerate(tickers)
    }

    logger.info(
        "Tickers: %s",
        tickers,
    )

    logger.info(
        "Numeric features: %s",
        len(FEATURES),
    )

    logger.info(
        "Total model input features: %s",
        len(FEATURES) + len(tickers),
    )

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
        medians,
        ticker_to_index,
    )


# ============================================================
# SEQUENCE GENERATION
# ============================================================

def create_sequences(
    frame,
    scaled_features,
    ticker_to_index,
    sequence_length,
    context=None,
    context_scaled=None,
):

    """
    Creates chronological sequences independently for each ticker.

    context:
        Previous split rows used only to supply historical
        sequence context.

    Targets are generated only for the current frame.
    """

    X_sequences = []
    y_values = []
    metadata = []

    ticker_count = len(ticker_to_index)

    for ticker in sorted(
        frame["Ticker"].unique()
    ):

        current = frame[
            frame["Ticker"] == ticker
        ].copy()

        current_indices = current.index.to_numpy()

        current_scaled = scaled_features[
            current_indices
        ]

        current_target = current[
            TARGET
        ].to_numpy(dtype=float)

        current_dates = current[
            "Date"
        ].to_numpy()

        # Optional historical context from previous split.
        if context is not None:

            context_ticker = context[
                context["Ticker"] == ticker
            ].copy()

            if len(context_ticker) > 0:

                context_indices = (
                    context_ticker.index.to_numpy()
                )

                context_values = context_scaled[
                    context_indices
                ]

                combined_features = np.concatenate(
                    [
                        context_values,
                        current_scaled,
                    ],
                    axis=0,
                )

                context_length = len(
                    context_values
                )

            else:

                combined_features = current_scaled

                context_length = 0

        else:

            combined_features = current_scaled

            context_length = 0

        ticker_index = ticker_to_index.get(
            ticker
        )

        if ticker_index is None:

            logger.warning(
                "Skipping unknown ticker: %s",
                ticker,
            )

            continue

        # One-hot ticker encoding.
        ticker_encoding = np.zeros(
            ticker_count,
            dtype=np.float32,
        )

        ticker_encoding[
            ticker_index
        ] = 1.0

        ticker_encoding = np.tile(
            ticker_encoding,
            (len(combined_features), 1),
        )

        combined_features = np.concatenate(
            [
                combined_features,
                ticker_encoding,
            ],
            axis=1,
        )

        # Generate targets only for current split.
        for position in range(
            len(current)
        ):

            combined_position = (
                context_length + position
            )

            start_position = (
                combined_position
                - sequence_length
                + 1
            )

            if start_position < 0:

                continue

            sequence = combined_features[
                start_position:
                combined_position + 1
            ]

            if len(sequence) != sequence_length:

                continue

            target = current_target[
                position
            ]

            if not np.isfinite(target):

                continue

            X_sequences.append(
                sequence.astype(np.float32)
            )

            y_values.append(
                np.float32(target)
            )

            metadata.append(
                {
                    "Date": current_dates[position],
                    "Ticker": ticker,
                    TARGET: target,
                }
            )

    if not X_sequences:

        raise ValueError(
            "No sequences generated. "
            "Check sequence length and dataset sizes."
        )

    X = np.asarray(
        X_sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        y_values,
        dtype=np.float32,
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    logger.info(
        "Sequence dataset shape: %s",
        X.shape,
    )

    logger.info(
        "Target shape: %s",
        y.shape,
    )

    return X, y, metadata_df


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_model(
    model_type,
    input_shape,
):

    model = Sequential(
        name=f"{model_type}_stock_predictor"
    )

    model.add(
        Input(
            shape=input_shape
        )
    )

    if model_type == "LSTM":

        model.add(
            LSTM(
                32,
                return_sequences=False,
            )
        )

    elif model_type == "GRU":

        model.add(
            GRU(
                32,
                return_sequences=False,
            )
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_type}"
        )

    model.add(
        Dropout(0.20)
    )

    model.add(
        Dense(
            16,
            activation="relu",
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:

        raise ValueError(
            "No valid observations for evaluation."
        )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    nonzero = actual != 0

    if nonzero.sum() > 0:

        directional_accuracy = np.mean(
            np.sign(actual[nonzero])
            == np.sign(predicted[nonzero])
        )

    else:

        directional_accuracy = np.nan

    zero_predictions = np.zeros_like(
        actual
    )

    zero_rmse = np.sqrt(
        mean_squared_error(
            actual,
            zero_predictions,
        )
    )

    zero_mae = mean_absolute_error(
        actual,
        zero_predictions,
    )

    return {
        "Observations": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy,
        "Zero_Baseline_MAE": zero_mae,
        "Zero_Baseline_RMSE": zero_rmse,
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_type,
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    logger.info(
        "Starting %s training.",
        model_type,
    )

    model = build_model(
        model_type,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        ),
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
        ],
        shuffle=False,
        verbose=1,
    )

    validation_predictions = model.predict(
        X_validation,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    logger.info(
        "%s validation RMSE: %.8f",
        model_type,
        validation_metrics["RMSE"],
    )

    return (
        model,
        history,
        validation_metrics,
        validation_predictions,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info(
        "Starting LSTM and GRU experiments."
    )

    (
        train,
        validation,
        test,
    ) = load_all_data()

    (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
        medians,
        ticker_to_index,
    ) = prepare_data(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # TRAIN SEQUENCES
    # --------------------------------------------------------

    X_train, y_train, train_meta = create_sequences(
        train,
        train_scaled,
        ticker_to_index,
        SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------
    # VALIDATION SEQUENCES
    # Use the last 60 train rows as historical context.
    # Validation targets remain validation-only.
    # --------------------------------------------------------

    X_validation, y_validation, validation_meta = (
        create_sequences(
            validation,
            validation_scaled,
            ticker_to_index,
            SEQUENCE_LENGTH,
            context=train,
            context_scaled=train_scaled,
        )
    )

    # --------------------------------------------------------
    # TEST SEQUENCES
    # Use validation rows as historical context.
    # Test targets remain test-only.
    # --------------------------------------------------------

    X_test, y_test, test_meta = create_sequences(
        test,
        test_scaled,
        ticker_to_index,
        SEQUENCE_LENGTH,
        context=validation,
        context_scaled=validation_scaled,
    )

    logger.info(
        "Training sequences: %s",
        X_train.shape,
    )

    logger.info(
        "Validation sequences: %s",
        X_validation.shape,
    )

    logger.info(
        "Test sequences: %s",
        X_test.shape,
    )

    # --------------------------------------------------------
    # TRAIN LSTM AND GRU
    # --------------------------------------------------------

    results = []
    trained_models = {}

    for model_type in [
        "LSTM",
        "GRU",
    ]:

        (
            model,
            history,
            validation_metrics,
            validation_predictions,
        ) = train_model(
            model_type,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        trained_models[
            model_type
        ] = model

        model_path = (
            MODEL_DIR
            / f"{model_type.lower()}_stock_model.keras"
        )

        model.save(
            model_path
        )

        logger.info(
            "Saved %s model: %s",
            model_type,
            model_path,
        )

        result = {
            "Model": model_type,
            **validation_metrics,
        }

        results.append(
            result
        )

    # --------------------------------------------------------
    # VALIDATION COMPARISON
    # --------------------------------------------------------

    validation_results = pd.DataFrame(
        results
    )

    validation_results = validation_results.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    validation_path = (
        REPORT_DIR
        / "deep_learning_validation_results.csv"
    )

    validation_results.to_csv(
        validation_path,
        index=False,
    )

    print("\n" + "=" * 90)

    print(
        "DEEP LEARNING VALIDATION COMPARISON"
    )

    print("=" * 90)

    print(
        validation_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    best_model_type = validation_results.iloc[
        0
    ]["Model"]

    best_model = trained_models[
        best_model_type
    ]

    logger.info(
        "Selected model: %s",
        best_model_type,
    )

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    test_predictions = best_model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    test_results = pd.DataFrame(
        [
            {
                "Selected_Model": best_model_type,
                **test_metrics,
            }
        ]
    )

    test_path = (
        REPORT_DIR
        / "deep_learning_test_results.csv"
    )

    test_results.to_csv(
        test_path,
        index=False,
    )

    print("\n" + "=" * 90)

    print(
        "SELECTED DEEP LEARNING MODEL — TEST RESULTS"
    )

    print("=" * 90)

    print(
        test_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_df = test_meta.copy()

    prediction_df[
        "Predicted_Return"
    ] = test_predictions

    prediction_df[
        "Prediction_Error"
    ] = (
        prediction_df[TARGET]
        - prediction_df["Predicted_Return"]
    )

    prediction_df[
        "Actual_Direction"
    ] = np.sign(
        prediction_df[TARGET]
    )

    prediction_df[
        "Predicted_Direction"
    ] = np.sign(
        prediction_df["Predicted_Return"]
    )

    prediction_path = (
        REPORT_DIR
        / "deep_learning_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE PREPROCESSING METADATA
    # --------------------------------------------------------

    metadata = {
        "selected_model": best_model_type,
        "features": FEATURES,
        "target": TARGET,
        "sequence_length": SEQUENCE_LENGTH,
        "ticker_to_index": ticker_to_index,
        "training_medians": medians.to_dict(),
        "scaler": scaler,
        "random_seed": SEED,
    }

    metadata_path = (
        MODEL_DIR
        / "deep_learning_metadata.joblib"
    )

    joblib.dump(
        metadata,
        metadata_path,
    )

    logger.info(
        "Saved preprocessing metadata: %s",
        metadata_path,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 90)

    print(
        "DEEP LEARNING PIPELINE COMPLETED"
    )

    print("=" * 90)

    print(
        f"Selected model: {best_model_type}"
    )

    print(
        f"Validation report: {validation_path}"
    )

    print(
        f"Test report: {test_path}"
    )

    print(
        f"Test predictions: {prediction_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )

    logger.info(
        "Deep learning experiments completed."
    )


if __name__ == "__main__":

    main()
