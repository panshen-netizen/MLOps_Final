"""
prepare_hmda_model_clean_data.py

Purpose
-------
Clean the HMDA EDA-ready dataset before feature engineering.

Input
-----
data/processed/hmda_eda_ready.csv

Output
------
data/processed/hmda_model_clean.csv

Main logic
----------
1. Load EDA-ready HMDA data.
2. Drop action_taken because loan_approved is already created.
3. Remove invalid credit score type codes 11-15.
4. Remove Not_applicable records from selected categorical fields.
5. Keep Exempt for selected categorical fields.
6. Remove missing / unknown conforming loan limit values.
7. Treat selected zero values as missing and remove them.
8. Keep valid zero values in selected numeric fields.
9. Remove negative and missing values from numeric columns.
10. Save feature-engineering-ready data.

"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_eda_ready.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hmda_model_clean.csv"


# ============================================================
# 2. Column groups
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

# 0 should be treated as missing for these columns
ZERO_AS_MISSING_COLUMNS = [
    "tract_median_age_of_housing_units",
    "tract_owner_occupied_units",
]

# 0 can be kept for these columns
ZERO_CAN_KEEP_COLUMNS = [
    "tract_minority_population_percent",
    "tract_population",
    "income",
]

# Remove Not_applicable from these categorical variables
REMOVE_NOT_APPLICABLE_COLUMNS = [
    "co_applicant_age",
    "applicant_age",
    "loan_purpose",
]

# Remove invalid credit score type codes 11-15
CREDIT_SCORE_TYPE_COLUMNS = [
    "co_applicant_credit_score_type",
    "applicant_credit_score_type",
]

DROP_MISSING_COLUMNS = [
    "conforming_loan_limit",
    "state_code",
    "county_code",
]


# ============================================================
# 3. Main cleaning pipeline
# ============================================================

def main() -> None:

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Please run prepare_hmda_eda_data.py first."
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("Input shape:", df.shape)

    # ------------------------------------------------------------
    # 1. Drop action_taken
    # ------------------------------------------------------------
    
    # loan_approved is already created, so action_taken is no longer needed.

    df = df.drop(columns=["action_taken"])

    print("Shape after dropping action_taken:", df.shape)
    print()

    # ------------------------------------------------------------
    # 2. Standardize text values
    # ------------------------------------------------------------

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert string-like missing values to np.nan.
    # both U and NA are treated as missing and will be removed later.
    general_missing_values = [
        "",
        " ",
        "NA",
        "N/A",
        "na",
        "n/a",
        "U",
        "u",
        "NULL",
        "null",
        "None",
        "none",
        "nan",
        "NaN",
    ]

    df = df.replace(general_missing_values, np.nan)

    # ------------------------------------------------------------
    # 3. Convert numeric columns
    # ------------------------------------------------------------

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------
    # 4. Remove invalid credit score type codes 11-15
    # ------------------------------------------------------------

    # Exempt and No_co_applicant are kept.

    invalid_credit_score_codes = [
        "11", "12", "13", "14", "15",
        11, 12, 13, 14, 15
    ]

    for col in CREDIT_SCORE_TYPE_COLUMNS:
        before = len(df)
        df = df[~df[col].isin(invalid_credit_score_codes)].copy()
        after = len(df)
        print(f"Removed invalid 11-15 from {col}: {before - after} rows")

    # ------------------------------------------------------------
    # 5. Remove Not_applicable from selected categorical columns
    # ------------------------------------------------------------

    for col in REMOVE_NOT_APPLICABLE_COLUMNS:
        before = len(df)
        df = df[df[col] != "Not_applicable"].copy()
        after = len(df)
        print(f"Removed Not_applicable from {col}: {before - after} rows")

    # ------------------------------------------------------------
    # 6. Keep Exempt for selected categorical columns
    # ------------------------------------------------------------

    # reverse_mortgage: Exempt kept
    # business_or_commercial_purpose: Exempt kept
    # open_end_line_of_credit: Exempt kept
    # applicant_credit_score_type: Exempt kept
    # co_applicant_credit_score_type: Exempt kept
    
    # No action is needed here.

    # ------------------------------------------------------------
    # 7. Remove rows with missing values in selected categorical columns
    # ------------------------------------------------------------

    # NaN, U, and NA are treated as missing and removed.

    before = len(df)
    df = df.dropna(subset=DROP_MISSING_COLUMNS).copy()
    after = len(df)

    print(f"Removed rows with missing selected categorical values: {before - after}")

    # ------------------------------------------------------------
    # 8. Treat selected 0 values as missing and remove them
    # ------------------------------------------------------------

    # tract_median_age_of_housing_units = 0 -> missing
    # tract_owner_occupied_units = 0 -> missing

    for col in ZERO_AS_MISSING_COLUMNS:
        before = len(df)
        df = df[df[col] != 0].copy()
        after = len(df)
        print(f"Removed {col} == 0 rows: {before - after}")

    # ------------------------------------------------------------
    # 9. Keep selected 0 values
    # ------------------------------------------------------------

    # tract_minority_population_percent, tract_population, and income can keep 0.
    # No action is needed here.

    # ------------------------------------------------------------
    # 10. Remove negative values and missing values from numeric columns
    # ------------------------------------------------------------

    # Negative values are invalid for these numeric variables.

    for col in NUMERIC_COLUMNS:
        before = len(df)
        df = df[df[col] >= 0].copy()
        after = len(df)
        print(f"Removed negative or missing rows from {col}: {before - after}")

    print()

    # The line above also removes NaN because NaN >= 0 is False.
    # This satisfies: numeric negative values and missing values are removed.

    # ------------------------------------------------------------
    # 11. Final output checks
    # ------------------------------------------------------------

    print("\nFinal shape:", df.shape)

    print("\nRemaining missing values:")

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]

    if len(missing_summary) == 0:
        print("No remaining missing values.")
    else:
        print(missing_summary.to_string())

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nModel-clean dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()