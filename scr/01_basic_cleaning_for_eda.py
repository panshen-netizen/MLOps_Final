"""
prepare_hmda_eda_data.py

Purpose
-------
Prepare raw HMDA 2024 data for EDA.

Main logic
----------
1. Load raw HMDA data.
2. Keep only approved and denied records.
3. Create a binary target variable: loan_approved.
4. Drop unused or leakage-risk columns.
5. Standardize column names.
6. Normalize special HMDA missing values for EDA.
7. Create has_co_applicant.
8. Standardize final output data types.

Output
------
data/processed/hmda_eda_ready.csv

Notes
-----
This script prepares data for EDA.
It does not train a model.
It does not impute missing values.
"""

import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "HMDA_2024.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "hmda_eda_ready.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Columns to drop
# ============================================================

DROP_COLUMNS = [
    "activity_year",
    "lei",
    "census_tract",
    "derived_loan_product_type",
    "derived_dwelling_category",
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
    "purchaser_type",
    "interest_rate",
    "rate_spread",
    "hoepa_status",
    "total_loan_costs",
    "total_points_and_fees",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "prepayment_penalty_term",
    "intro_rate_period",
    "negative_amortization",
    "interest_only_payment",
    "balloon_payment",
    "other_nonamortizing_features",
    "construction_method",
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "total_units",
    "multifamily_affordable_units",
    "debt_to_income_ratio",
    "applicant_ethnicity-1",
    "applicant_ethnicity-2",
    "applicant_ethnicity-3",
    "applicant_ethnicity-4",
    "applicant_ethnicity-5",
    "co-applicant_ethnicity-1",
    "co-applicant_ethnicity-2",
    "co-applicant_ethnicity-3",
    "co-applicant_ethnicity-4",
    "co-applicant_ethnicity-5",
    "applicant_ethnicity_observed",
    "co-applicant_ethnicity_observed",
    "applicant_race-1",
    "applicant_race-2",
    "applicant_race-3",
    "applicant_race-4",
    "applicant_race-5",
    "co-applicant_race-1",
    "co-applicant_race-2",
    "co-applicant_race-3",
    "co-applicant_race-4",
    "co-applicant_race-5",
    "applicant_race_observed",
    "co-applicant_race_observed",
    "applicant_sex",
    "co-applicant_sex",
    "applicant_sex_observed",
    "co-applicant_sex_observed",
    "applicant_age_above_62",
    "co-applicant_age_above_62",
    "submission_of_application",
    "initially_payable_to_institution",
    "aus-1",
    "aus-2",
    "aus-3",
    "aus-4",
    "aus-5",
    "denial_reason-1",
    "denial_reason-2",
    "denial_reason-3",
    "denial_reason-4",
    "tract_one_to_four_family_homes",
    "loan_to_value_ratio",
]


# ============================================================
# 3. Final numeric columns
# ============================================================

NUMERIC_COLUMNS = [
    "tract_median_age_of_housing_units",
    "tract_owner_occupied_units",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
    "tract_minority_population_percent",
    "tract_population",
    "income",
    "property_value",
    "loan_term",
    "loan_amount",
]


# ============================================================
# 4. Helper functions
# ============================================================

def standardize_column_name(col: str) -> str:
    """
    Convert raw HMDA column names to snake_case.
    """
    return (
        col.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def replace_if_exists(df: pd.DataFrame, col: str, mapping: dict) -> None:
    """
    Replace values only if the column exists."""
    if col in df.columns:
        df[col] = df[col].replace(mapping)


def standardize_general_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert general missing-like values to np.nan.
    conforming_loan_limit is excluded because NA is a valid category.
    """

    general_missing_values = [
        "",
        " ",
        "NA",
        "N/A",
        "na",
        "n/a",
        "NULL",
        "null",
        "None",
        "none",
    ]

    for col in df.columns:
        if col == "conforming_loan_limit":
            continue

        df[col] = df[col].replace(general_missing_values, np.nan)

    return df


# ============================================================
# 5. Main pipeline
# ============================================================

def main() -> None:

    # ------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}\n"
            "Please update RAW_DATA_PATH if your file name is different."
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    print("Raw data shape:", df.shape)

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # ------------------------------------------------------------
    # Keep only approve and deny records
    # ------------------------------------------------------------

    df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")

    df = df[df["action_taken"].isin([1, 2, 3])].copy()

    df["loan_approved"] = np.where(df["action_taken"].isin([1, 2]), 1, 0)

    print("Shape after keeping approved and denied records:", df.shape)

    # ------------------------------------------------------------
    # Drop columns
    # ------------------------------------------------------------

    existing_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
    missing_drop_columns = [col for col in DROP_COLUMNS if col not in df.columns]

    df = df.drop(columns=existing_drop_columns)

    print("Dropped columns:", len(existing_drop_columns))

    if missing_drop_columns:
        print("Columns not found and skipped:", len(missing_drop_columns))
        for col in missing_drop_columns:
            print("-", col)

    # ------------------------------------------------------------
    # Standardize column names
    # ------------------------------------------------------------

    df.columns = [standardize_column_name(col) for col in df.columns]

    print("\nColumn names standardized.")

    # ------------------------------------------------------------
    # Create has_co_applicant
    # ------------------------------------------------------------

    no_co_applicant = pd.Series(False, index=df.index)

    if "co_applicant_age" in df.columns:
        no_co_applicant = no_co_applicant | (
            df["co_applicant_age"].astype(str).str.strip() == "9999"
        )

    if "co_applicant_credit_score_type" in df.columns:
        no_co_applicant = no_co_applicant | (
            df["co_applicant_credit_score_type"].astype(str).str.strip() == "10"
        )

    df["has_co_applicant"] = (~no_co_applicant).astype(int)

    # ------------------------------------------------------------
    # Standardize general missing values
    # ------------------------------------------------------------

    df = standardize_general_missing_values(df)

    # ------------------------------------------------------------
    # Normalize special HMDA values for EDA
    # ------------------------------------------------------------

    # These two numeric fields use 0 as missing.
    replace_if_exists(
        df,
        "ffiec_msa_md_median_family_income",
        {"0": np.nan, 0: np.nan},
    )

    replace_if_exists(
        df,
        "tract_to_msa_income_percentage",
        {"0": np.nan, 0: np.nan},
    )

    # These numeric fields use Exempt as missing.
    replace_if_exists(
        df,
        "loan_term",
        {
            "Exempt": np.nan,
            "exempt": np.nan,
            "EXEMPT": np.nan,
        },
    )

    replace_if_exists(
        df,
        "property_value",
        {
            "Exempt": np.nan,
            "exempt": np.nan,
            "EXEMPT": np.nan,
        },
    )

    # Age fields are categorical.
    # 8888 and 9999 are converted to readable labels.
    age_mapping = {
        "8888": "Not_applicable",
        8888: "Not_applicable",
        "9999": "No_co_applicant",
        9999: "No_co_applicant",
    }

    replace_if_exists(df, "applicant_age", age_mapping)
    replace_if_exists(df, "co_applicant_age", age_mapping)

    # 1111 = Exempt for selected categorical fields.
    exempt_columns = [
        "reverse_mortgage",
        "open_end_line_of_credit",
        "business_or_commercial_purpose",
        "co_applicant_credit_score_type",
        "applicant_credit_score_type",
    ]

    for col in exempt_columns:
        replace_if_exists(
            df,
            col,
            {
                "1111": "Exempt",
                1111: "Exempt",
            },
        )

    # co_applicant_credit_score_type:
    # 10 = No co-applicant
    replace_if_exists(
        df,
        "co_applicant_credit_score_type",
        {
            "10": "No_co_applicant",
            10: "No_co_applicant",
        },
    )

    # ------------------------------------------------------------
    # Standardize final output types
    # ------------------------------------------------------------

    # Target columns.
    df["loan_approved"] = df["loan_approved"].astype(int)

    if "action_taken" in df.columns:
        df["action_taken"] = df["action_taken"].astype(int)

    # These columns are numeric.
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 8888, 9999, and 11111 in numeric columns are not replaced.
    # If they appear in numeric columns, they remain numeric values.

    # All other are categorical.
    non_numeric_columns = [
        col for col in df.columns
        if col not in NUMERIC_COLUMNS + ["loan_approved", "action_taken"]
    ]

    for col in non_numeric_columns:
        df[col] = df[col].astype("string")

    # ------------------------------------------------------------
    # Output checks
    # ------------------------------------------------------------

    print("\nFinal EDA dataset shape:", df.shape)

    print("\nNumeric columns:")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            print("-", col)

    print("\nCategorical columns:")
    for col in non_numeric_columns:
        print("-", col)

    print("\nFinal data types:")
    print(df.dtypes.to_string())

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nEDA-ready dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()