# dags/cv_quality_gate_dag.py
# DAG 4: Quality Gate Workflow (Chạy trước khi Full Rollout)

from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import timedelta

# Cấu hình chung cho Kubernetes (Sử dụng config CV Factory)
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
    dag_id="cv_quality_gate_workflow",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None, # Chỉ chạy khi được trigger bởi DAG khác (ví dụ: sau Canary Check)
    catchup=False,
    default_args={
        "owner": "mlops-qa",
        "retries": 0, 
        "execution_timeout": timedelta(hours=1)
    },
    tags=["cv", "quality-gate", "sla"],
) as dag:
    
    # Lấy tham số URI/VERSION đang được kiểm tra
    MODEL_URI_TO_CHECK = "{{ dag_run.conf['model_uri'] if dag_run and dag_run.conf and dag_run.conf['model_uri'] else 'models:/defect_detector/canary' }}"
    MODEL_NAME = "defect_detector"
    
    # 1. Tác vụ: Kiểm tra Chất lượng Mô hình (Role EXPERIMENTATION)
    # Chạy Evaluator trên một mẫu dữ liệu Production (Hardening Quality Gate)
    check_model_health = KubernetesPodOperator(
        task_id="check_model_health_on_live_data",
        name="model-health-check",
        image="your-registry/cv-factory:experimentation-v1.0",
        cmds=["python3"],
        arguments=["scripts/check_model_health.py", 
                   "--config", "configs/evaluator/prod_eval_config.json", 
                   "--model-uri", MODEL_URI_TO_CHECK],
        **K8S_FULL_CONFIG
    )

    # 2. Tác vụ: Kiểm tra SLA/Độ trễ (Role EXPERIMENTATION)
    run_sla_load_test = KubernetesPodOperator(
        task_id="run_sla_load_test",
        name="sla-load-test-job",
        image="your-registry/cv-factory:experimentation-v1.0",
        cmds=["python3"],
        arguments=["scripts/run_load_test.py", "--endpoint", "http://cv-inference-service:80", "--users", "100"],
        **K8S_FULL_CONFIG
    )
    
    # 3. Tác vụ: Kiểm tra Sức khỏe Dịch vụ (Liveness/Readiness/Model Load)
    monitor_full_service_health = KubernetesPodOperator(
        task_id="monitor_full_service_health",
        name="service-health-monitor",
        image="your-registry/cv-factory:etl-v1.0", # Sử dụng ETL Role cho monitor
        cmds=["python3"],
        arguments=["scripts/monitor_service_health.py", "--endpoint-url", "http://cv-inference-service:80"],
        **K8S_FULL_CONFIG
    )

    # 4. Tác vụ: Trigger Full Rollout (Nếu tất cả các kiểm tra PASS)
    trigger_full_rollout = TriggerDagRunOperator(
        task_id="trigger_full_rollout_dag",
        trigger_dag_id="cv_deployment_canary_workflow", # Giả định DAG này có logic full rollout
        conf={"model_uri": MODEL_URI_TO_CHECK, "mode": "full_rollout"},
        trigger_rule="all_success",
    )


    # Định nghĩa Luồng (Flow)
    # 3 kiểm tra chạy song song, sau đó Trigger Full Rollout
    [check_model_health, run_sla_load_test, monitor_full_service_health] >> trigger_full_rollout