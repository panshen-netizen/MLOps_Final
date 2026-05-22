"""
gradio_app.py

Purpose
-------
A simple web UI for HMDA loan approval model deployment.

Function
--------
1. Upload a CSV file.
2. Run model inference.
3. Generate prediction results and metrics.
4. First uploaded file is saved as reference validation data.
5. Later uploaded files are treated as changed data.
6. Evidently automatically generates:

"""

import os
import json
import contextlib
import h2o
import pandas as pd
import gradio as gr

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

WEB_OUTPUT_DIR = OUTPUT_DIR / "web_app_predictions"
WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PREDICTIONS_PATH = WEB_OUTPUT_DIR / "reference_predictions.csv"
CURRENT_PREDICTIONS_PATH = WEB_OUTPUT_DIR / "current_predictions.csv"

REFERENCE_EVIDENTLY_HTML_PATH = WEB_OUTPUT_DIR / "reference_evidently_report.html"
DRIFT_EVIDENTLY_HTML_PATH = WEB_OUTPUT_DIR / "drift_evidently_report.html"

REFERENCE_EVIDENTLY_JSON_PATH = WEB_OUTPUT_DIR / "reference_evidently_report.json"
DRIFT_EVIDENTLY_JSON_PATH = WEB_OUTPUT_DIR / "drift_evidently_report.json"


# ============================================================
# 2. Settings
# ============================================================

TARGET = "loan_approved"
CUSTOM_THRESHOLD = 0.65

COLUMNS_TO_MONITOR = [
    TARGET,
    "loan_amount",
    "property_value",
    "income",
    "loan_to_value_ratio",
    "interest_rate",
    "loan_purpose",
    "occupancy_type",
    "property_type",
    "predicted_probability_approved",
    "predicted_label_custom_threshold",
]

# ============================================================
# 3. H2O model loading
# ============================================================

model = None


def start_h2o_quietly():
    """
    Start H2O while hiding long startup messages.
    """

    h2o.no_progress()

    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            h2o.init(
                max_mem_size="4G",
                nthreads=-1,
            )


def find_latest_h2o_model(model_dir: Path) -> Path:
    """
    Find the latest saved H2O model in the models folder.
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidate_files = [
        path
        for path in model_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]

    if len(candidate_files) == 0:
        raise FileNotFoundError(f"No saved H2O model found in: {model_dir}")

    return max(candidate_files, key=lambda path: path.stat().st_mtime)


def load_model_once():
    """
    Load the H2O model only once.
    """

    global model

    if model is None:
        start_h2o_quietly()
        model_path = find_latest_h2o_model(MODEL_DIR)
        model = h2o.load_model(str(model_path))
        print("Loaded H2O model:")
        print(model_path)

    return model


# ============================================================
# 4. Prediction helpers
# ============================================================

def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Evaluate binary classification predictions.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob),
    }

    return metrics


def run_batch_prediction(df: pd.DataFrame):
    """
    Run H2O model prediction on uploaded CSV data.
    """

    loaded_model = load_model_once()

    has_target = TARGET in df.columns

    if has_target:
        y_true = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
        feature_df = df.drop(columns=[TARGET])
    else:
        y_true = None
        feature_df = df.copy()

    input_h2o = h2o.H2OFrame(feature_df)

    predictions_h2o = loaded_model.predict(input_h2o)
    predictions = predictions_h2o.as_data_frame()

    if "p1" not in predictions.columns:
        raise ValueError("Prediction output does not contain p1 probability column.")

    y_prob = pd.to_numeric(predictions["p1"], errors="coerce")
    y_pred = (y_prob >= CUSTOM_THRESHOLD).astype(int)

    output_df = df.copy()
    output_df["predicted_probability_approved"] = y_prob
    output_df["predicted_label_custom_threshold"] = y_pred
    output_df["custom_threshold"] = CUSTOM_THRESHOLD

    prediction_rate = pd.Series(y_pred).value_counts(normalize=True).to_dict()

    summary = {
        "rows": len(df),
        "custom_threshold": CUSTOM_THRESHOLD,
        "target_column_found": has_target,
        "predicted_approved_rate": float(prediction_rate.get(1, 0)),
        "predicted_denied_rate": float(prediction_rate.get(0, 0)),
        "average_predicted_approval_probability": float(y_prob.mean()),
    }

    if has_target:
        summary["metrics"] = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

    return output_df, summary


# ============================================================
# 5. Evidently helpers
# ============================================================

def prepare_evidently_columns(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """
    Keep common monitoring columns.
    """

    available_columns = [
        col
        for col in COLUMNS_TO_MONITOR
        if col in reference_df.columns and col in current_df.columns
    ]

    if len(available_columns) == 0:
        raise ValueError("No common columns found for Evidently monitoring.")

    return (
        reference_df[available_columns].copy(),
        current_df[available_columns].copy(),
        available_columns,
    )


def run_reference_evidently_report(reference_df: pd.DataFrame):
    """
    Generate a baseline Evidently report for the first uploaded dataset.
    """

    reference_data = reference_df.copy()

    try:
        from evidently import Report
        from evidently.presets import DataSummaryPreset

        report = Report([
            DataSummaryPreset(),
        ])

        result = report.run(
            current_data=reference_data,
        )

        result.save_html(str(REFERENCE_EVIDENTLY_HTML_PATH))

        with open(REFERENCE_EVIDENTLY_JSON_PATH, "w") as f:
            json.dump(result.dict(), f, indent=2, default=str)

    except Exception as error:
        print("New Evidently API failed. Trying old API.")
        print("Error:", error)

        from evidently.report import Report
        from evidently.metric_preset import DataQualityPreset

        report = Report(metrics=[
            DataQualityPreset(),
        ])

        report.run(
            current_data=reference_data,
        )

        report.save_html(str(REFERENCE_EVIDENTLY_HTML_PATH))

        with open(REFERENCE_EVIDENTLY_JSON_PATH, "w") as f:
            json.dump(report.as_dict(), f, indent=2, default=str)

    return str(REFERENCE_EVIDENTLY_HTML_PATH)


def run_drift_evidently_report():
    """
    Generate Evidently drift report comparing reference vs current.
    """

    reference_df = pd.read_csv(REFERENCE_PREDICTIONS_PATH, low_memory=False)
    current_df = pd.read_csv(CURRENT_PREDICTIONS_PATH, low_memory=False)

    reference_data, current_data, monitored_columns = prepare_evidently_columns(
        reference_df=reference_df,
        current_df=current_df,
    )

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        report = Report([
            DataDriftPreset(),
            DataSummaryPreset(),
        ])

        result = report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        result.save_html(str(DRIFT_EVIDENTLY_HTML_PATH))

        with open(DRIFT_EVIDENTLY_JSON_PATH, "w") as f:
            json.dump(result.dict(), f, indent=2, default=str)

    except Exception as error:
        print("New Evidently API failed. Trying old API.")
        print("Error:", error)

        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        report.save_html(str(DRIFT_EVIDENTLY_HTML_PATH))

        with open(DRIFT_EVIDENTLY_JSON_PATH, "w") as f:
            json.dump(report.as_dict(), f, indent=2, default=str)

    return str(DRIFT_EVIDENTLY_HTML_PATH), monitored_columns


# ============================================================
# 6. Gradio app logic
# ============================================================

def reset_demo():
    """
    Reset reference/current files and reports.
    """

    files_to_delete = [
        REFERENCE_PREDICTIONS_PATH,
        CURRENT_PREDICTIONS_PATH,
        REFERENCE_EVIDENTLY_HTML_PATH,
        DRIFT_EVIDENTLY_HTML_PATH,
        REFERENCE_EVIDENTLY_JSON_PATH,
        DRIFT_EVIDENTLY_JSON_PATH,
    ]

    deleted = []

    for path in files_to_delete:
        if path.exists():
            path.unlink()
            deleted.append(str(path))

    return "Demo reset completed. Upload validation data first."


def process_uploaded_file(file):
    """
    Main single-entry upload function.
    """

    if file is None:
        return (
            "Please upload a CSV file.",
            None,
            None,
            None,
        )

    file_path = Path(file.name)

    if file_path.suffix.lower() != ".csv":
        return (
            "Please upload a CSV file.",
            None,
            None,
            None,
        )

    df = pd.read_csv(file_path, low_memory=False)

    prediction_df, summary = run_batch_prediction(df)

    # First upload becomes reference.
    if not REFERENCE_PREDICTIONS_PATH.exists():
        dataset_role = "Reference Validation Data"
        prediction_output_path = REFERENCE_PREDICTIONS_PATH
        prediction_df.to_csv(prediction_output_path, index=False)

        evidently_report_path = run_reference_evidently_report(prediction_df)

        message = (
            "Prediction completed successfully.\n"
            "Baseline Evidently report generated successfully."
        )

    else:
        dataset_role = "Current Changed Data"
        prediction_output_path = CURRENT_PREDICTIONS_PATH
        prediction_df.to_csv(prediction_output_path, index=False)

        evidently_report_path, monitored_columns = run_drift_evidently_report()

        message = (
            "Prediction completed successfully.\n"
            "Evidently drift report generated successfully."
        )

    metrics = summary.get("metrics", {})

    metrics_text = f"""
## Prediction Summary

| Item | Value |
|---|---:|
| Dataset Role | {dataset_role} |
| Rows | {summary.get("rows", "N/A"):,} |
| Threshold | {summary.get("custom_threshold", "N/A")} |
| Target Column Found | {summary.get("target_column_found", "N/A")} |
| Predicted Approved Rate | {summary.get("predicted_approved_rate", 0):.2%} |
| Predicted Denied Rate | {summary.get("predicted_denied_rate", 0):.2%} |
| Average Approval Probability | {summary.get("average_predicted_approval_probability", 0):.2%} |

## Model Metrics

| Metric | Value |
|---|---:|
| Accuracy | {metrics.get("accuracy", 0):.4f} |
| Precision | {metrics.get("precision", 0):.4f} |
| Recall | {metrics.get("recall", 0):.4f} |
| F1-score | {metrics.get("f1", 0):.4f} |
| AUC | {metrics.get("auc", 0):.4f} |
| Log Loss | {metrics.get("logloss", 0):.4f} |
"""

    return (
        message,
        metrics_text,
        str(prediction_output_path),
        evidently_report_path,
    )

# ============================================================
# 7. Build UI
# ============================================================

with gr.Blocks(title="HMDA Loan Approval Model Demo") as demo:

    gr.Markdown(
        """
        # HMDA Loan Approval Model Demo

        Upload a CSV file to run batch inference.
        
        """
    )

    with gr.Row():
        file_input = gr.File(
            label="Upload CSV file",
            file_types=[".csv"],
        )

    with gr.Row():
        run_button = gr.Button("Run Prediction and Monitoring", variant="primary")
        reset_button = gr.Button("Reset")

    message_output = gr.Textbox(
        label="Status",
        lines=2,
        interactive=False,
    )

    metrics_output = gr.Markdown(
        label="Prediction Metrics and Summary"
    )

    prediction_file_output = gr.File(
        label="Download Prediction File",
    )

    evidently_file_output = gr.File(
        label="Download Evidently Report",
    )

    run_button.click(
        fn=process_uploaded_file,
        inputs=file_input,
        outputs=[
            message_output,
            metrics_output,
            prediction_file_output,
            evidently_file_output,
        ],
    )

    reset_button.click(
        fn=reset_demo,
        inputs=None,
        outputs=message_output,
    )


if __name__ == "__main__":
    load_model_once()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )