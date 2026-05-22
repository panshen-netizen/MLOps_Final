"""
hmda_training_dag.py

Purpose
-------
Airflow DAG for the HMDA model training pipeline.

Pipeline steps
--------------
1. Basic cleaning for EDA.
2. Prepare cleaned HMDA model data.
3. Create feature-engineered training data and reserved inference validation data.
4. Train H2O AutoML model with MLflow tracking.

"""

from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


# ============================================================
# 1. Project settings
# ============================================================

PROJECT_ROOT = "/Users/panshen/Desktop/MLOps_Final"
PYTHON_PATH = "/opt/anaconda3/envs/mlops_env/bin/python"


# ============================================================
# 2. DAG definition
# ============================================================

with DAG(
    dag_id="hmda_training_pipeline",
    description="HMDA data preparation and H2O AutoML training pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["hmda", "training", "h2o", "automl", "mlflow"],
) as dag:

    basic_cleaning_for_eda = BashOperator(
        task_id="01_basic_cleaning_for_eda",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_PATH}" scr/01_basic_cleaning_for_eda.py'
        ),
    )

    prepare_model_clean_data = BashOperator(
        task_id="02_prepare_hmda_model_clean_data",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_PATH}" scr/02_prepare_hmda_model_clean_data.py'
        ),
    )

    create_hmda_features = BashOperator(
        task_id="03_hmda_features",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_PATH}" scr/03_hmda_features.py'
        ),
    )

    train_h2o_automl = BashOperator(
        task_id="04_model_train_h2o_automl",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_PATH}" scr/04_model_train_h2o_automl.py'
        ),
    )

    (
        basic_cleaning_for_eda
        >> prepare_model_clean_data
        >> create_hmda_features
        >> train_h2o_automl
    )