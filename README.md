# MLOps_Final: HMDA Loan Approval Prediction Project

## 1. Data Source

This project uses the **2024 HMDA mortgage lending dataset**.

HMDA stands for **Home Mortgage Disclosure Act**. 

It is a public loan-level dataset reported by financial institutions and contains information about mortgage loan applications, loan characteristics, applicant information, and final loan outcomes.

## 2. Project Objective

The goal of this project is to predict whether a mortgage loan application will be **approved or denied**.

This is a **binary classification task**.

Target variable:

| Variable | Meaning |
|---|---|
| `loan_approved` | Whether the loan application was approved |

Target values:

| Value | Meaning |
|---|---|
| 1 | Approved |
| 0 | Denied |


## 3. Raw Data and Selected Columns

The original HMDA dataset contains many loan-level reporting fields.  
For this project, only the most relevant variables were kept for modeling and inference.

Selected columns:

- loan_approved
- loan_amount
- property_value
- income
- loan_to_value_ratio
- interest_rate
- loan_purpose
- occupancy_type
- property_type

## 4. Variable Meaning

| Variable                           | Meaning                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `loan_approved`                    | Target variable showing whether the application was approved or denied         |
| `loan_amount`                      | Requested mortgage loan amount                                                 |
| `property_value`                   | Reported value of the property                                                 |
| `income`                           | Applicant income                                                               |
| `loan_to_value_ratio`              | Ratio between loan amount and property value                                   |
| `interest_rate`                    | Interest rate of the loan                                                      |
| `loan_purpose`                     | Purpose of the mortgage loan                                                   |
| `occupancy_type`                   | How the property will be occupied                                              |
| `property_type`                    | Type of property                                                               |
| `predicted_probability_approved`   | Model-predicted probability of approval                                        |
| `predicted_label_custom_threshold` | Final predicted label after applying the custom threshold                      |
| `custom_threshold`                 | Probability threshold used to convert probability into approved / denied label |

# HMDA Loan Approval Prediction Project

## 1. Data Source

This project uses the **2024 HMDA mortgage lending dataset**.

HMDA stands for **Home Mortgage Disclosure Act**. HMDA data are public loan-level mortgage application records reported by financial institutions. The dataset includes information about mortgage loan applications, applicant characteristics, loan features, property information, and final loan outcomes.

---

## 2. Project Objective

The goal of this project is to predict whether a mortgage loan application will be **approved or denied**.

This is a **binary classification task**.

Target variable:

| Variable | Meaning |
|---|---|
| `loan_approved` | Whether the mortgage loan application was approved |

Target values:

| Value | Meaning |
|---|---|
| 1 | Approved |
| 0 | Denied |

---

## 3. Raw Data and Selected Columns

The original HMDA dataset contains many loan-level reporting fields.  
For this project, we selected a smaller set of variables that are relevant to mortgage approval prediction.

Selected modeling columns:

| Column | Used For |
|---|---|
| `loan_approved` | Target variable |
| `loan_amount` | Model feature |
| `property_value` | Model feature |
| `income` | Model feature |
| `loan_to_value_ratio` | Engineered model feature |
| `interest_rate` | Model feature |
| `loan_purpose` | Model feature |
| `occupancy_type` | Model feature |
| `property_type` | Model feature |

Model-generated output columns:

| Column | Meaning |
|---|---|
| `predicted_probability_approved` | Predicted probability that the loan is approved |
| `predicted_label_custom_threshold` | Final prediction after applying the custom threshold |
| `custom_threshold` | Threshold used to convert probability into a class label |

---

## 4. Variable Meaning

| Variable | Meaning |
|---|---|
| `loan_approved` | Target variable showing whether the application was approved or denied |
| `loan_amount` | Requested mortgage loan amount |
| `property_value` | Reported value of the property |
| `income` | Applicant income |
| `loan_to_value_ratio` | Ratio between loan amount and property value |
| `interest_rate` | Interest rate of the loan |
| `loan_purpose` | Purpose of the mortgage loan |
| `occupancy_type` | How the property will be occupied |
| `property_type` | Type of property |
| `predicted_probability_approved` | Model-predicted probability of approval |
| `predicted_label_custom_threshold` | Final predicted label after applying the custom threshold |
| `custom_threshold` | Probability threshold used to convert probability into approved / denied label |

---

## 5. Categorical Variable Meaning

### `loan_approved`

| Value | Meaning |
|---|---|
| 1 | Approved |
| 0 | Denied |

### `loan_purpose`

| Value | Meaning |
|---|---|
| 1 | Home purchase |
| 2 | Home improvement |
| 31 | Refinancing |
| 32 | Cash-out refinancing |
| 4 | Other purpose |
| 5 | Not applicable |

### `occupancy_type`

| Value | Meaning |
|---|---|
| 1 | Principal residence |
| 2 | Second residence |
| 3 | Investment property |

### `property_type`

| Value | Meaning |
|---|---|
| 1 | One-to-four family property |
| 2 | Manufactured home |
| 3 | Multifamily property |

## 6. Project Pipeline

### Step 1: Exploratory Data Analysis

**File:** `EDA/EDA.ipynb`

**Purpose:** Explore the HMDA dataset, inspect missing values, check variable distributions, and understand the loan approval outcome.

---

### Step 2: Basic Cleaning for EDA

**File:** `scr/01_basic_cleaning_for_eda.py`

**Purpose:** Clean the raw HMDA data for exploratory data analysis.

---

### Step 3: Prepare Model-Clean Data

**File:** `scr/02_prepare_hmda_model_clean_data.py`

**Purpose:** Prepare the cleaned modeling dataset and create the binary target variable `loan_approved`.

---

### Step 4: Feature Engineering

**File:** `scr/03_hmda_features.py`

**Purpose:** Create model-ready features, including `loan_to_value_ratio`, and prepare training and validation datasets.

---

### Step 5: Model Training with H2O AutoML

**File:** `scr/04_model_train_h2o_automl.py`

**Purpose:** Train a binary classification model using H2O AutoML, select the best model, evaluate it, and save model outputs.

**Main outputs:**

- `outputs/h2o_automl_leaderboard.csv`
- `outputs/h2o_test_metrics.txt`
- `models/`
- `mlruns/`

---

### Step 6: Create Changed Inference Data

**File:** `scr/06_create_changed_inference_data.py`

**Purpose:** Create a changed version of the validation dataset by modifying at least two feature values.

**Changed features:**

- `income`
- `loan_to_value_ratio`

**Purpose of the change:** The changed dataset is used to test whether the deployed model and monitoring system can detect changes in input data and prediction behavior.

---

### Step 7: Web App Deployment and Inference

**File:** `scr/gradio_app.py`

**Purpose:** Deploy the trained model through a simple Gradio web application. Users can upload a CSV file, run batch inference, view prediction metrics, download prediction results, and generate Evidently reports.

The first uploaded file is treated as **reference validation data**.

Later uploaded files are treated as **current changed data**.

---

### Step 8: Airflow Training Pipeline

**File:** `airflow_dag/hmda_training_dag.py`

**Purpose:** Use Airflow to orchestrate the model training workflow.

**Pipeline structure:**

1. Prepare model data
2. Run feature engineering
3. Train H2O AutoML model
4. Save model outputs

---

## 7. Tools Used

| Tool | Role in This Project |
|---|---|
| GitHub | Stores project code, scripts, notebooks, README, and documentation |
| lakeFS | Versions raw data, cleaned data, feature-engineered data, validation data, and changed inference data |
| Airflow | Orchestrates the data preparation and model training pipeline |
| H2O AutoML | Automatically trains and compares multiple machine learning models |
| MLflow | Tracks model parameters, metrics, leaderboard files, model artifacts, and experiment outputs |
| Gradio | Provides a simple web interface for model inference and file upload |
| Evidently | Generates baseline reports and drift monitoring reports |

---

## 8. Model Training

The model is trained using **H2O AutoML**.

H2O AutoML automatically compares multiple model families and selects the best-performing model based on evaluation metrics.

Main evaluation metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Among predicted approved applications, how many were actually approved |
| Recall | Among actually approved applications, how many were correctly identified |
| F1-score | Balance between precision and recall |
| AUC | Ability of the model to distinguish approved and denied applications |
| Log Loss | Error of predicted probabilities |

Training outputs are saved in:

- `outputs/`
- `models/`
- `mlruns/`

---

## 9. Model Deployment

The trained model is deployed through a **Gradio web app**.

The web app allows users to:

- Upload a validation CSV file
- Run batch prediction
- View prediction metrics
- Download prediction results
- Download Evidently reports

The web app file is:

- `scr/gradio_app.py`

---

## 10. Model Monitoring

Model monitoring is performed using **Evidently**.

The first uploaded validation dataset is used as the **reference dataset**.

Later uploaded changed datasets are used as **current datasets**.

Evidently compares:

- Reference validation data
- Current changed data

The monitoring report checks changes in important features and prediction outputs, including:

- `income`
- `loan_to_value_ratio`
- `predicted_probability_approved`
- `predicted_label_custom_threshold`

Two types of reports are generated:

| Report | Meaning |
|---|---|
| Baseline Evidently report | Generated after uploading the original validation data |
| Drift Evidently report | Generated after uploading the changed validation data |

---

## 11. Data Versioning with lakeFS

lakeFS is used to version data files.

The following data stages are versioned:

- Raw HMDA data
- Cleaned HMDA data
- Feature-engineered training data
- Validation data
- Changed inference data

This helps track which data version was used for:

- Training
- Validation
- Inference
- Model monitoring

---

## 12. Experiment Tracking with MLflow

MLflow is used to track model experiments and outputs.

MLflow records:

- Model parameters
- Training metrics
- H2O AutoML leaderboard
- Saved model artifacts
- Prediction outputs
- Experiment logs

MLflow UI can be opened with:

`mlflow ui --backend-store-uri ./mlruns --port 5001`
