# infra_deployment/dags/retraining_dag.py (HARDENED)
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from .base_dag_config import DEFAULT_DAG_ARGS, SCRIPTS_DIR, CONFIG_DIR, PYTHON_EXEC_COMMAND

# --- Configuration ---
RETRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, 'retraining_main.py')
RETRAIN_CONFIG = os.path.join(CONFIG_DIR, 'retrain_config.yaml')

# NOTE: Giảm số lần retry để tránh gửi Job Training nhiều lần
retraining_args = DEFAULT_DAG_ARGS.copy()
retraining_args['retries'] = 0 

with DAG(
    dag_id='medical_imaging_retraining_check',
    default_args=retraining_args,
    schedule_interval='0 0 * * *', # Chạy định kỳ hàng ngày
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'retraining', 'trigger'],
) as dag:
    
    # Task 1: Check Triggers and Submit Job
    # RetrainOrchestrator bên trong script sẽ:
    # 1. Chạy tất cả các Triggers (Drift, Performance, Time).
    # 2. Nếu kích hoạt, nó sẽ gọi job_utils.submit_training_job (không cần logic phức tạp trong DAG).
    check_and_submit_job = BashOperator(
        task_id='check_and_submit_retrain_job',
        bash_command=(
            f"{PYTHON_EXEC_COMMAND} {RETRAIN_SCRIPT} "
            f"--config-path {RETRAIN_CONFIG} "
            f"--mode check_and_submit" # Tham số này chỉ thị cho script driver
        ),
    )