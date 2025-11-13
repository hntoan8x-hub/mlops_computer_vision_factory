# dags/cv_deployment_dag.py
# DAG 3: Triển khai An toàn (Canary Rollout & Validation Workflow)

from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import timedelta
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

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
    dag_id="cv_deployment_canary_workflow",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None, # Chỉ chạy khi được trigger thủ công hoặc bởi DAG khác
    catchup=False,
    default_args={
        "owner": "mlops-deployment",
        "retries": 0, 
        "execution_timeout": timedelta(hours=1)
    },
    tags=["cv", "deployment", "canary"],
) as dag:
    
    # Lấy tham số version mới từ conf (được truyền từ trigger)
    # Giả định tham số URI được truyền từ Training Job
    NEW_MODEL_URI = "{{ dag_run.conf['model_uri'] if dag_run and dag_run.conf and dag_run.conf['model_uri'] else 'models:/defect_detector/latest' }}"
    MODEL_NAME = "defect_detector"

    # 1. Tác vụ: Triển khai phiên bản mới (STANDARD DEPLOYMENT cho phiên bản MỚI)
    deploy_new_version = KubernetesPodOperator(
        task_id="deploy_new_version",
        name="deploy-new-version",
        image="your-registry/cv-factory:experimentation-v1.0",
        cmds=["python3"],
        arguments=["scripts/deploy_standard.py", 
                   "--config", "configs/deployment/deployment_config.json",
                   "--uri", NEW_MODEL_URI, 
                   "--name", MODEL_NAME],
        **K8S_FULL_CONFIG
    )

    # 2. Tác vụ: Kiểm thử Tải (Load Test - Role EXPERIMENTATION)
    run_load_test = KubernetesPodOperator(
        task_id="run_load_test_validation",
        name="load-test-job",
        image="your-registry/cv-factory:experimentation-v1.0",
        cmds=["python3"],
        arguments=["scripts/run_load_test.py", "--endpoint", "http://cv-inference-service:80"],
        **K8S_FULL_CONFIG
    )
    
    # 3. Tác vụ: Canary Rollout (Role EXPERIMENTATION)
    # Sử dụng script run_canary_rollout.py đã Hardened
    canary_rollout = KubernetesPodOperator(
        task_id="run_canary_rollout",
        name="canary-rollout-job",
        image="your-registry/cv-factory:experimentation-v1.0",
        cmds=["python3"],
        arguments=["scripts/run_canary_rollout.py", 
                   "--config", "configs/deployment/deployment_config.json",
                   "--uri", NEW_MODEL_URI, 
                   "--name", MODEL_NAME,
                   "--stable-version", "{{ var.value.get('stable_cv_version', 'v1.0.0') }}"], # Lấy stable version từ Airflow Variable
        # HARDENING: Nếu canary thất bại, nó sẽ SystemExit. Rollback sẽ được kích hoạt
        trigger_rule="all_done",
        **K8S_FULL_CONFIG
    )

    # 4. Tác vụ: Rollback nếu Canary thất bại (Luồng khẩn cấp)
    # Sử dụng script rollback_deployment.py đã Hardened
    rollback_on_failure = KubernetesPodOperator(
        task_id="rollback_deployment",
        name="rollback-job",
        image="your-registry/cv-factory:experimentation-v1.0",
        cmds=["python3"],
        arguments=["scripts/rollback_deployment.py", 
                   "--config", "configs/deployment/deployment_config.json",
                   "--target-version", "{{ var.value.get('stable_cv_version', 'v1.0.0') }}", # Rollback về stable version
                   "--name", MODEL_NAME],
        # CRITICAL: Chỉ chạy Rollback nếu tác vụ trước đó thất bại
        trigger_rule="one_failed", 
        **K8S_FULL_CONFIG
    )


    # Định nghĩa Luồng (Flow): Deploy -> Load Test -> Canary (thành công) HOẶC Rollback (thất bại)
    deploy_new_version >> run_load_test >> canary_rollout
    canary_rollout >> rollback_on_failure