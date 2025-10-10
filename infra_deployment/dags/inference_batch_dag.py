# infra_deployment/dags/inference_batch_dag.py (HARDENED)
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from .base_dag_config import DEFAULT_DAG_ARGS, SCRIPTS_DIR, CONFIG_DIR, PYTHON_EXEC_COMMAND

# --- Configuration ---
INFERENCE_BATCH_SCRIPT = os.path.join(SCRIPTS_DIR, 'inference_batch_main.py')
INFERENCE_CONFIG = os.path.join(CONFIG_DIR, 'medical_inference_batch_config.yaml')

with DAG(
    dag_id='medical_imaging_batch_scoring',
    default_args=DEFAULT_DAG_ARGS,
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'inference', 'batch'],
) as dag:

    # Task 1: Run the batch inference job
    # The script uses CVInferenceOrchestrator to read batch data, score it, and write the output.
    run_batch_scoring = BashOperator(
        task_id='run_daily_batch_inference',
        bash_command=(
            f"{PYTHON_EXEC_COMMAND} {INFERENCE_BATCH_SCRIPT} "
            f"--config-path {INFERENCE_CONFIG} "
            f"--run-date {{ ds }}" # Truyền ngày chạy Airflow cho mục đích kiểm toán dữ liệu
        ),
    )