# dags/cv_retrain_dag.py
# DAG 2: Quản lý Chất lượng Mô hình (TMR - Trigger, Monitor, Retrain)

from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import timedelta

# Cấu hình chung cho Kubernetes
K8S_FULL_CONFIG = {
    "namespace": "mlops-cv-factory",
    "image_pull_policy": "Always",
    "service_account_name": "airflow-k8s-sa",
    "startup_timeout_seconds": 600, 
    "env_from": [
        {"configMapRef": {"name": "cv-factory-configs"}},
        {"secretRef": {"name": "cv-factory-secrets"}}
    ]
}

with DAG(
    dag_id="cv_tmr_workflow",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="0 6 * * *", 
    catchup=False,
    default_args={
        "owner": "ai-governance",
        "retries": 0, 
        "execution_timeout": timedelta(hours=1) # Rút ngắn timeout vì TMR chỉ là check/submit
    },
    tags=["cv", "tmr", "governance"],
) as dag:

    # 1. Tác vụ: Chạy TMR Check (Role ETL)
    # Tác vụ này chạy TMRFacade.run_tmr_workflow()
    # Nếu trigger fire, RetrainOrchestrator TỰ SUBMIT job Training (tác vụ 2 chỉ là placeholder/monitor)
    check_tmr_triggers = KubernetesPodOperator(
        task_id="check_tmr_triggers_and_submit",
        name="tmr-checker",
        image="your-registry/cv-factory:etl-v1.0",
        cmds=["python3"],
        arguments=["scripts/run_retrain_check.py", "--config", "configs/monitoring/tmr_config.json"],
        # Chạy task này để nó thực hiện check và submit Job Training nếu cần
        trigger_rule="all_done",
        **K8S_FULL_CONFIG
    )

    # 2. Tác vụ: Giám sát Job Training (Giả định Job đã được submit ở bước 1)
    # Tác vụ này sẽ đợi Job Training Kubernetes/SageMaker hoàn thành
    monitor_training_job = BashOperator(
        task_id="monitor_submitted_training_job",
        bash_command="echo 'Waiting for Retraining Job to complete via MLOps API...' && sleep 120", # MOCK: Giả lập chờ
        trigger_rule="all_success", # Chạy tiếp theo
    )
    
    # 3. Tác vụ: Trigger Deployment Workflow (Kích hoạt DAG khác)
    # Job Training đã TỰ ĐỘNG trigger Deployment khi hoàn thành (nhờ CVTrainingOrchestrator)
    # Task này chỉ là MOCK/Placeholder/Notification
    notify_deployment_ready = BashOperator(
        task_id="notify_deployment_ready",
        bash_command="echo 'Training completed. Deployment is expected to be triggered by CVTrainingOrchestrator'",
        trigger_rule="all_success",
    )

    # Định nghĩa Luồng (Flow)
    check_tmr_triggers >> monitor_training_job >> notify_deployment_ready