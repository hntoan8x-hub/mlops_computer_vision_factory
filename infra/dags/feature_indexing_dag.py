# dags/feature_indexing_dag.py
# DAG 1: Indexing Feature Store Workflow (CV Factory)

from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import timedelta

# Cấu hình chung cho Kubernetes
K8S_FULL_CONFIG = {
    "namespace": "mlops-cv-factory", # Đổi namespace
    "image_pull_policy": "Always",
    "service_account_name": "airflow-k8s-sa",
    "startup_timeout_seconds": 600, 
    "env_from": [
        {"configMapRef": {"name": "cv-factory-configs"}}, # Đổi ConfigMap
        {"secretRef": {"name": "cv-factory-secrets"}} # Đổi Secret
    ]
}

with DAG(
    dag_id="cv_feature_indexing_workflow",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="0 3 * * *", 
    catchup=False,
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=3)
    },
    tags=["cv", "feature-store", "indexing"],
) as dag:

    # 1. Tác vụ: Kiểm tra Sức khỏe Feature Store DB
    check_db_health = KubernetesPodOperator(
        task_id="check_fs_db_health",
        name="check-fs-db-health",
        image="your-registry/cv-factory:etl-v1.0", # Dùng ETL Role cho job nhẹ
        cmds=["python3"],
        arguments=["scripts/monitor_service_health.py", "--endpoint-url", "$(FEATURE_STORE_URL)"], # Kiểm tra Endpoint FS
        **K8S_FULL_CONFIG
    )

    # 2. Tác vụ: Chạy Indexing Job (Được tích hợp vào Training Job)
    # Chúng ta sử dụng Training Job để chạy CVTrainingOrchestrator, nơi có bước lập chỉ mục Hardened
    run_indexing = KubernetesPodOperator(
        task_id="run_feature_indexing_job",
        name="feature-indexing-job",
        image="your-registry/cv-factory:trainer-v1.0", # Sử dụng TRAINER Role
        cmds=["python3"], # Override ENTRYPOINT để chạy Indexing Job riêng biệt nếu cần
        arguments=[
            "scripts/run_training_job.py", 
            "--config", "configs/pipelines/indexing_only_config.json", # Cấu hình chỉ chạy Indexing
            "--id", "fs_indexing_{{ ds_nodash }}"
        ],
        container_resources={
            "request_cpu": "2000m", 
            "limit_cpu": "4000m", 
            "limit_memory": "8Gi"
        },
        **K8S_FULL_CONFIG
    )

    # 3. Tác vụ: Dọn dẹp Artifact
    cleanup_old_data = KubernetesPodOperator(
        task_id="cleanup_old_artifacts",
        name="cleanup-job",
        image="your-registry/cv-factory:etl-v1.0",
        cmds=["python3"],
        arguments=["scripts/cleanup_artifacts.py", "--config", "configs/governance/cleanup_config.yaml", "--model-name", "defect_detector"],
        **K8S_FULL_CONFIG
    )

    # Định nghĩa Luồng (Flow)
    check_db_health >> run_indexing >> cleanup_old_data