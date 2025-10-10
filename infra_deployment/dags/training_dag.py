# infra_deployment/dags/training_dag.py (HARDENED)
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from .base_dag_config import DEFAULT_DAG_ARGS, SCRIPTS_DIR, CONFIG_DIR, PYTHON_EXEC_COMMAND

# --- Configuration ---
TRAINING_SCRIPT = os.path.join(SCRIPTS_DIR, 'training_main.py')
TRAINING_CONFIG = os.path.join(CONFIG_DIR, 'medical_training_config.yaml')

# Define DAG
with DAG(
    dag_id='medical_imaging_training_pipeline',
    default_args=DEFAULT_DAG_ARGS,
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'training', 'model_build'],
) as dag:

    # Task 1: Runs the training_main.py script.
    # The script uses CVPipelineFactory to handle all DI, Orchestration, Logging, and Registration.
    run_full_training_flow = BashOperator(
        task_id='run_full_mlops_training',
        bash_command=(
            f"{PYTHON_EXEC_COMMAND} {TRAINING_SCRIPT} "
            f"--config-path {TRAINING_CONFIG} "
            f"--run-id {{ ti.dag_run.run_id }}" # Truyền Run ID Airflow làm ID cho MLflow Run
        ),
    )