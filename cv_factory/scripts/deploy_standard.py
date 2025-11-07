# scripts/deploy_standard.py

import argparse
import logging
import os
import json
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
from shared_libs.orchestrators.cv_deployment_orchestrator import CVDeploymentOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError

# --- Import các MLOps Services MOCK/TEST ---
from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger as MockLogger
from shared_libs.monitoring.event_emitter import ConsoleEventEmitter as MockEmitter

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_DEPLOY_SCRIPT")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Tải cấu hình từ file JSON (hoặc YAML trong môi trường Production).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """
    Hàm chính thực thi luồng Standard Deployment.
    """
    parser = argparse.ArgumentParser(description="Kích hoạt CV Deployment Orchestrator ở chế độ STANDARD.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML của pipeline.")
    parser.add_argument("--uri", type=str, required=True, 
                        help="URI của Model Artifact đã đăng ký (ví dụ: models:/defect_detector/3).")
    parser.add_argument("--name", type=str, required=True, 
                        help="Tên mô hình (ví dụ: defect_detector).")
    parser.add_argument("--id", type=str, default="cv_standard_deploy_01", 
                        help="ID duy nhất cho lần chạy Orchestrator này.")
    args = parser.parse_args()

    try:
        # 1. Tải Cấu hình
        raw_config = load_config(args.config)
        
        # 2. Khởi tạo MLOps Services (MOCK/TEST)
        mlops_services = {
            "logger_service": MockLogger(),
            "event_emitter": MockEmitter(),
        }

        # 3. Lắp ráp và Khởi tạo Deployment Orchestrator
        logger.info(f"Instantiating CVDeploymentOrchestrator with ID: {args.id}.")
        deployment_orchestrator = CVDeploymentOrchestrator(
            orchestrator_id=args.id,
            config=raw_config,
            **mlops_services
        )

        # 4. THỰC THI WORKFLOW: STANDARD DEPLOYMENT
        logger.info(f"Starting STANDARD Deployment for model URI: {args.uri}")
        
        # Deployer sẽ tự động lấy version tag từ URI nếu cần
        endpoint_id = deployment_orchestrator.run(
            model_artifact_uri=args.uri,
            model_name=args.name,
            mode="standard" # <<< CHẾ ĐỘ TRIỂN KHAI >>>
        )

        # 5. Báo cáo Kết quả
        logger.info("=====================================================")
        logger.info("✅ DEPLOYMENT COMPLETED SUCCESSFULLY.")
        logger.info(f"✅ Endpoint ID: {endpoint_id}")
        logger.info("=====================================================")

    except FileNotFoundError as e:
        logger.error(f"❌ CONFIG ERROR: {e}")
        exit(1)
    except InvalidConfigError as e:
        logger.error(f"❌ VALIDATION ERROR: Configuration did not pass Pydantic schema check. Details: {e}")
        exit(1)
    except WorkflowExecutionError as e:
        logger.critical(f"❌ CRITICAL FAILURE: Deployment failed during execution. Details: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"❌ UNEXPECTED ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()