# scripts/run_training_job.py

import argparse
import logging
import json
import os
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
from shared_libs.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError

# --- Import các MLOps Services MOCK/TEST ---
# Trong môi trường thực tế, các lớp này sẽ là các instance REAL của MLflow, Kafka, v.v.
from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger as MockLogger
from shared_libs.ml_core.mlflow_service.implementations.mlflow_registry import MLflowRegistry as MockRegistry
from shared_libs.monitoring.event_emitter import ConsoleEventEmitter as MockEmitter

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_TRAINING_SCRIPT")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Tải cấu hình từ file JSON (hoặc YAML trong môi trường Production).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        # Giả định file config là JSON
        return json.load(f)

def main():
    """
    Hàm chính thực thi toàn bộ luồng Training -> Evaluation -> Registration -> Deployment.
    """
    parser = argparse.ArgumentParser(description="Kích hoạt CV Training Orchestrator.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML của pipeline.")
    parser.add_argument("--id", type=str, default="cv_training_run_01", 
                        help="ID duy nhất cho lần chạy Orchestrator này.")
    args = parser.parse_args()

    try:
        # 1. Tải và Xác thực Cấu hình
        raw_config = load_config(args.config)
        logger.info(f"Configuration loaded successfully from {args.config}.")
        
        # 2. Khởi tạo MLOps Services (MOCK/TEST)
        # Trong môi trường DDP, các service này thường chỉ được khởi tạo trên tiến trình chính (Rank 0)
        mlops_services = {
            "logger_service": MockLogger(), # MLflow Tracker
            "registry_service": MockRegistry(), # MLflow Registry
            "event_emitter": MockEmitter(), # Console Event Emitter
        }

        # 3. Lắp ráp và Khởi tạo Orchestrator Cốt lõi
        logger.info(f"Instantiating CVTrainingOrchestrator with ID: {args.id}.")
        training_orchestrator = CVTrainingOrchestrator(
            orchestrator_id=args.id,
            config=raw_config,
            **mlops_services
        )

        # 4. THỰC THI WORKFLOW
        logger.info("Starting End-to-End Training Workflow...")
        final_metrics, model_uri = training_orchestrator.run()

        # 5. Báo cáo Kết quả
        logger.info("=====================================================")
        logger.info(f"✅ WORKFLOW COMPLETED SUCCESSFULLY. Metrics: {final_metrics}")
        logger.info(f"✅ Model Registered URI: {model_uri}")
        logger.info("=====================================================")

    except FileNotFoundError as e:
        logger.error(f"❌ CONFIG ERROR: {e}")
        exit(1)
    except InvalidConfigError as e:
        logger.error(f"❌ VALIDATION ERROR: Configuration did not pass Pydantic schema check. Details: {e}")
        exit(1)
    except WorkflowExecutionError as e:
        logger.critical(f"❌ CRITICAL FAILURE: Workflow failed during execution. Details: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"❌ UNEXPECTED ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()