"""
model_train_h2o_automl.py

Purpose
-------
Train a binary classification model using H2O AutoML and MLflow.

Input
-----
data/processed/hmda_features_train.csv

Output
------
outputs/h2o_automl_leaderboard.csv
outputs/h2o_test_metrics.txt
models/
mlruns/

Main logic
----------
1. Load feature-engineered HMDA data.
2. Randomly sample 200,000 rows for faster training.
3. Split data into train and test.
4. Train H2O AutoML.
5. Evaluate the leader model on the internal test set.
6. Save the leaderboard, metrics, and trained leader model.
7. Log parameters, metrics, and artifacts to MLflow.

"""

import os
import warnings
import logging
import contextlib
import mlflow
import pandas as pd
import h2o

from h2o.automl import H2OAutoML
from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_features_train.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

LEADERBOARD_PATH = OUTPUT_DIR / "h2o_automl_leaderboard.csv"
METRICS_PATH = OUTPUT_DIR / "h2o_test_metrics.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Training settings
# ============================================================

TRAIN_SAMPLE_SIZE = 300_000
RANDOM_STATE = 42

TARGET = "loan_approved"


# ============================================================
# 3. Suppress unnecessary logs
# ============================================================

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)


# ============================================================
# 4. Helper functions
# ============================================================

def extract_h2o_metric(metric_result):
    """
    Convert H2O metric output to a single float value.
    """

    if metric_result is None:
        return None

    if isinstance(metric_result, list):
        if len(metric_result) == 0:
            return None

        first_item = metric_result[0]

        if isinstance(first_item, list):
            if len(first_item) >= 2:
                return float(first_item[1])
            if len(first_item) == 1:
                return float(first_item[0])

        return float(first_item)

    return float(metric_result)


def log_mlflow_metric(metric_name, metric_value):
    """
    Log a metric to MLflow only if the metric is valid.
    """

    if metric_value is None:
        return

    mlflow.log_metric(metric_name, float(metric_value))


def evaluate_h2o_model(model, frame, dataset_name):
    """
    Evaluate an H2O binary classification model.
    """

    perf = model.model_performance(frame)

    metrics = {
        f"{dataset_name}_auc": extract_h2o_metric(perf.auc()),
        f"{dataset_name}_accuracy": extract_h2o_metric(perf.accuracy()),
        f"{dataset_name}_precision": extract_h2o_metric(perf.precision()),
        f"{dataset_name}_recall": extract_h2o_metric(perf.recall()),
        f"{dataset_name}_f1": extract_h2o_metric(perf.F1()),
        f"{dataset_name}_logloss": extract_h2o_metric(perf.logloss()),
        f"{dataset_name}_mean_per_class_error": extract_h2o_metric(
            perf.mean_per_class_error()
        ),
    }

    return metrics, perf


def start_h2o_quietly():
    """
    Start H2O while hiding long startup messages.
    """

    h2o.no_progress()

    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            h2o.init(
                max_mem_size="6G",
                nthreads=-1,
            )

    print("H2O initialized successfully.")


# ============================================================
# 5. Main training pipeline
# ============================================================

def main() -> None:

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature data not found: {DATA_PATH}\n"
            "Please run prepare_hmda_features.py first."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Input data shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

    # ------------------------------------------------------------
    # Randomly sample 200,000 rows for training
    # ------------------------------------------------------------

    if len(df) > TRAIN_SAMPLE_SIZE:
        df = df.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    print("Data shape after sampling:", df.shape)

    # ------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("HMDA_H2O_AutoML")

    with mlflow.start_run(run_name="h2o_automl_training_200k"):

        mlflow.log_param("dataset_path", str(DATA_PATH))
        mlflow.log_param("target", TARGET)
        mlflow.log_param("sample_size", len(df))
        mlflow.log_param("sampling_strategy", "random_sample_200k")
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_param("automl_max_models", 20)
        mlflow.log_param("automl_sort_metric", "AUC")
        mlflow.log_param("balance_classes", True)
        mlflow.log_param("nfolds", 5)

        # ------------------------------------------------------------
        # Start H2O
        # ------------------------------------------------------------

        start_h2o_quietly()

        # ------------------------------------------------------------
        # Convert pandas DataFrame to H2OFrame
        # ------------------------------------------------------------

        hf = h2o.H2OFrame(df)

        # Target must be categorical for binary classification.
        hf[TARGET] = hf[TARGET].asfactor()

        features = [col for col in hf.columns if col != TARGET]

        print("\nTarget:", TARGET)
        print("Number of features:", len(features))

        mlflow.log_param("number_of_features", len(features))

        # ------------------------------------------------------------
        # Train-test split
        # ------------------------------------------------------------

        train, test = hf.split_frame(
            ratios=[0.8],
            seed=RANDOM_STATE,
        )

        print("Train rows:", train.nrows)
        print("Test rows:", test.nrows)

        mlflow.log_param("train_rows", train.nrows)
        mlflow.log_param("test_rows", test.nrows)
        mlflow.log_param("train_test_split", "80/20")

        # ------------------------------------------------------------
        # Train H2O AutoML
        # ------------------------------------------------------------

        aml = H2OAutoML(
            max_models=20,
            seed=RANDOM_STATE,
            sort_metric="AUC",
            balance_classes=True,
            nfolds=5,
        )

        aml.train(
            x=features,
            y=TARGET,
            training_frame=train,
        )

        # ------------------------------------------------------------
        # Save and log AutoML leaderboard
        # ------------------------------------------------------------

        leaderboard = aml.leaderboard.as_data_frame()
        leaderboard.to_csv(LEADERBOARD_PATH, index=False)

        print()
        print("AutoML Leaderboard:")
        print(leaderboard.head(10).to_string(index=False))

        mlflow.log_artifact(str(LEADERBOARD_PATH))

        # ------------------------------------------------------------
        # Evaluate leader model on internal test set
        # ------------------------------------------------------------

        leader = aml.leader

        print()
        print("Leader model id:")
        print(leader.model_id)

        mlflow.log_param("leader_model_id", leader.model_id)
        mlflow.log_param("leader_algorithm", leader.algo)

        test_metrics, test_perf = evaluate_h2o_model(
            model=leader,
            frame=test,
            dataset_name="test",
        )

        print()
        print("Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(metric_name, ":", metric_value)
            log_mlflow_metric(metric_name, metric_value)

        confusion_matrix = test_perf.confusion_matrix()
        confusion_matrix_str = str(confusion_matrix)

        # ------------------------------------------------------------
        # Save test metrics
        # ------------------------------------------------------------

        with open(METRICS_PATH, "w") as f:
            f.write("H2O AutoML Test Metrics\n")
            f.write("=======================\n\n")
            f.write("Script purpose: model training only\n")
            f.write("Sampling strategy: random sample 200,000 rows\n")
            f.write("Train-test split: 80/20\n")
            f.write("Balance classes in H2O: True\n")
            f.write("Inference validation dataset: not used in this script\n\n")

            f.write(f"Leader model id: {leader.model_id}\n")
            f.write(f"Leader algorithm: {leader.algo}\n\n")

            for metric_name, metric_value in test_metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")

            f.write("\nConfusion Matrix:\n")
            f.write(confusion_matrix_str)

        print()
        print("Metrics saved to:")
        print(METRICS_PATH)

        mlflow.log_artifact(str(METRICS_PATH))

        # ------------------------------------------------------------
        # Save trained leader model
        # ------------------------------------------------------------

        saved_model_path = h2o.save_model(
            model=leader,
            path=str(MODEL_DIR),
            force=True,
        )

        print()
        print("Leader model saved to:")
        print(saved_model_path)

        mlflow.log_artifact(saved_model_path)

        # ------------------------------------------------------------
        # Shutdown H2O
        # ------------------------------------------------------------

        h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()