# infra_deployment/dags/monitoring_dag.py (HARDENED)
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from .base_dag_config import DEFAULT_DAG_ARGS, SCRIPTS_DIR, CONFIG_DIR, PYTHON_EXEC_COMMAND
from datetime import timedelta

# --- Configuration ---
MONITORING_SCRIPT = os.path.join(SCRIPTS_DIR, 'monitoring_main.py')
MONITORING_CONFIG = os.path.join(CONFIG_DIR, 'monitoring_config.yaml')

# NOTE: Giả sử lịch chạy kiểm tra hàng giờ (important for real-time monitoring)
with DAG(
    dag_id='mlops_monitoring_pipeline',
    default_args=DEFAULT_DAG_ARGS,
    schedule_interval='0 * * * *', # Chạy mỗi giờ (at minute 0)
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'monitoring', 'drift_check'],
) as dag:

    # Task 1: Runs the monitoring_main.py script.
    # Script này sẽ kích hoạt MonitoringOrchestrator để:
    # 1. Chạy tất cả các Monitors (Drift, Fairness, Latency).
    # 2. Báo cáo (Report) và gửi cảnh báo (Alert) nếu cần.
    run_monitoring_check = BashOperator(
        task_id='run_all_monitoring_checks',
        bash_command=(
            f"{PYTHON_EXEC_COMMAND} {MONITORING_SCRIPT} "
            f"--config-path {MONITORING_CONFIG} "
            f"--check-time {{ ds }} {{ ti.execution_date.strftime('%H:%M:%S') }}" # Pass execution time for audit
        ),
        # Giảm thời gian retry để cảnh báo nhanh hơn
        retries=0 
    )