# infra_deployment/dags/deployment_dag.py (HARDENED)
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from .base_dag_config import DEFAULT_DAG_ARGS, SCRIPTS_DIR, CONFIG_DIR, PYTHON_EXEC_COMMAND

# --- Configuration ---
DEPLOYMENT_SCRIPT = os.path.join(SCRIPTS_DIR, 'deployment_main.py')
DEPLOYMENT_CONFIG = os.path.join(CONFIG_DIR, 'deployment_config.yaml')
TRAINING_DAG_ID = 'medical_imaging_training_pipeline' # Reference to the source DAG

default_args = DEFAULT_DAG_ARGS.copy()
# NOTE: Cần retry nếu thất bại vì Deployment là bước CRITICAL
default_args['retries'] = 2 

with DAG(
    dag_id='model_deployment_automation',
    default_args=default_args,
    schedule_interval=None, # Thường chạy 'manually' hoặc sau khi Training DAG hoàn thành
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'deployment', 'cd'],
) as dag:
    
    # Task 1 (Optional): Wait for the CV Training Pipeline to succeed and register a model
    # Note: Đây là một Dummy Task đơn giản để minh họa sự phụ thuộc.
    wait_for_model_registration = DummyOperator(
        task_id='wait_for_model_registration',
        # Trong thực tế: Dùng ExternalTaskSensor để chờ Training DAG (hoặc sự kiện MLflow Webhook)
    )

    # Task 2: Check Quality Gate and Deploy
    # Script này sẽ kiểm tra các điều kiện cuối cùng (ví dụ: Final metrics > 0.9)
    # và gọi Deployer Adapter để triển khai model lên Cloud.
    deploy_model_to_staging = BashOperator(
        task_id='deploy_model_to_staging_endpoint',
        bash_command=(
            f"{PYTHON_EXEC_COMMAND} {DEPLOYMENT_SCRIPT} "
            f"--config-path {DEPLOYMENT_CONFIG} "
            f"--stage staging"
        ),
        pool='deployment_pool' # Sử dụng Airflow Pool để giới hạn tài nguyên Deployment
    )
    
    # Task 3 (Optional): Run Post-Deployment Smoke Test (Kiểm tra nhanh)
    run_smoke_test = BashOperator(
        task_id='run_staging_smoke_test',
        bash_command="python $SCRIPTS_DIR/test_endpoint.py --endpoint-url $STAGING_URL",
        # Nếu Smoke Test thất bại, quá trình sẽ dừng lại (không chuyển traffic)
    )
    
    # Task 4: Switch Traffic (Go-Live)
    # Task này sẽ chạy lại script deployment_main.py với tham số khác (switch)
    # để chuyển lưu lượng truy cập 100% sang phiên bản mới.
    switch_to_production = BashOperator(
        task_id='go_live_production_switch',
        bash_command=(
            f"{PYTHON_EXEC_COMMAND} {DEPLOYMENT_SCRIPT} "
            f"--config-path {DEPLOYMENT_CONFIG} "
            f"--stage production_switch" 
        ),
    )

    # Định nghĩa luồng công việc
    wait_for_model_registration >> deploy_model_to_staging >> run_smoke_test >> switch_to_production