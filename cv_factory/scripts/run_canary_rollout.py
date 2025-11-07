# scripts/run_canary_rollout.py

import argparse
import logging
import os
import json
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
from shared_libs.orchestrators.cv_deployment_orchestrator import CVDeploymentOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError
from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger as MockLogger
from shared_libs.monitoring.event_emitter import ConsoleEventEmitter as MockEmitter

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_CANARY_SCRIPT")

def load_config(config_path: str) -> Dict[str, Any]:
    """Tải cấu hình từ file JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """
    Hàm chính thực thi luồng Canary Deployment.
    """
    parser = argparse.ArgumentParser(description="Kích hoạt CV Deployment Orchestrator ở chế độ CANARY.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML.")
    parser.add_argument("--uri", type=str, required=True, 
                        help="URI của Model Artifact đã đăng ký (phiên bản MỚI).")
    parser.add_argument("--stable-version", type=str, required=True,
                        help="Phiên bản ổn định hiện tại (để biết nên chuyển lưu lượng từ đâu).")
    parser.add_argument("--canary-percent", type=int, default=5,
                        help="Phần trăm lưu lượng truy cập ban đầu cho Canary (ví dụ: 5).")
    parser.add_argument("--id", type=str, default="cv_canary_rollout_01", 
                        help="ID duy nhất cho lần chạy Orchestrator này.")
    args = parser.parse_args()

    try:
        raw_config = load_config(args.config)
        
        mlops_services = {
            "logger_service": MockLogger(),
            "event_emitter": MockEmitter(),
        }

        deployment_orchestrator = CVDeploymentOrchestrator(
            orchestrator_id=args.id,
            config=raw_config,
            **mlops_services
        )

        # THỰC THI WORKFLOW: CANARY DEPLOYMENT
        logger.info(f"Starting CANARY Rollout for {args.uri}. Initial traffic: {args.canary_percent}%.")
        
        endpoint_id = deployment_orchestrator.run(
            model_artifact_uri=args.uri,
            model_name=raw_config['model']['name'], # Tên model/endpoint
            mode="canary", # <<< CHẾ ĐỘ TRIỂN KHAI >>>
            new_version_tag=args.uri.split('/')[-1], # Lấy tên version từ URI (ví dụ: '3' hoặc 'canary')
            stable_version=args.stable_version,
            canary_traffic_percent=args.canary_percent 
        )

        logger.info("=====================================================")
        logger.info("✅ CANARY ROLLOUT COMPLETED SUCCESSFULLY.")
        logger.info(f"✅ Endpoint ID: {endpoint_id}. Traffic is now split at {args.canary_percent}%.")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Canary rollout failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()