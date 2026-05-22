"""
prepare_hmda_features.py

Purpose
-------
Create a feature-engineered HMDA dataset.

Input
-----
data/processed/hmda_model_clean.csv

Output
------
data/processed/hmda_features_train.csv
data/inference_validation/hmda_inference_validation.csv

Main logic
----------
1. Apply log transformation to loan_amount, property_value, and income.
2. Calculate loan_to_value_ratio using cleaned loan_amount and property_value.
3. Drop preapproval and reverse_mortgage columns.
4. Randomly sample 10,000 rows for inference validation.
5. Save training data and inference validation data separately.

"""

import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_model_clean.csv"

TRAIN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_features_train.csv"
INFERENCE_VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "inference_validation" / "hmda_inference_validation.csv"
)


# ============================================================
# 2. Main feature engineering pipeline
# ============================================================

def main() -> None:

    # ------------------------------------------------------------
    # Load model-clean data
    # ------------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Please run prepare_hmda_model_clean_data.py first."
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("Input shape:", df.shape)

    # ------------------------------------------------------------
    # 1. Log transform selected numeric variables
    # ------------------------------------------------------------
    # log1p(x) = log(1 + x), which is safe when x = 0.
    # These variables are right-skewed based on EDA.

    df["log_loan_amount"] = np.log1p(df["loan_amount"])
    df["log_property_value"] = np.log1p(df["property_value"])
    df["log_income"] = np.log1p(df["income"])

    # ------------------------------------------------------------
    # 2. Calculate loan_to_value_ratio
    # ------------------------------------------------------------

    df["loan_to_value_ratio"] = (
        df["loan_amount"] / df["property_value"] * 100
    )

    # Cap LTV to avoid unreasonable values.
    df["loan_to_value_ratio"] = df["loan_to_value_ratio"].clip(
        lower=0,
        upper=200,
    )

    # ------------------------------------------------------------
    # 3. Drop columns not used for modeling
    # ------------------------------------------------------------

    columns_to_drop = ["preapproval", "reverse_mortgage",]

    df = df.drop(columns=columns_to_drop, errors="ignore")   

    # ------------------------------------------------------------
    # 4. Randomly sample 10,000 rows for inference validation
    # ------------------------------------------------------------
    # This dataset will not be used for training.
    # It will be used later to test deployed model inference.

    inference_validation_df = df.sample(
        n=10_000,
        random_state=42,
    )

    train_df = df.drop(index=inference_validation_df.index).copy()

    # ------------------------------------------------------------
    # 5. Final checks
    # ------------------------------------------------------------

    print("Full feature-engineered shape:", df.shape)
    print("Training feature data shape:", train_df.shape)
    print("Inference validation data shape:", inference_validation_df.shape)

    print()
    print("New engineered features:")
    print("- log_loan_amount")
    print("- log_property_value")
    print("- log_income")
    print("- loan_to_value_ratio")

    print()
    print("Dropped column:")
    print("- preapproval")
    print("- reverse_mortgage")

    print()
    print("Inference validation sample:")
    print("- 10,000 rows randomly sampled")
    print("- random_state = 42")

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    inference_validation_df.to_csv(
        INFERENCE_VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print()
    print("Training feature dataset saved to:")
    print(TRAIN_OUTPUT_PATH)

    print()
    print("Inference validation dataset saved to:")
    print(INFERENCE_VALIDATION_OUTPUT_PATH)


if __name__ == "__main__":
    main()