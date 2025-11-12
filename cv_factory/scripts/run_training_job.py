# scripts/run_training_job.py (HARDENED)

import argparse
import logging
import json
import os
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
# LOẠI BỎ: from shared_libs.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError

# <<< NEW: SỬ DỤNG PIPELINE RUNNER >>>
from shared_libs.orchestrators.pipeline_runner import PipelineRunner 

# LOẠI BỎ: Tất cả các imports MLOps Mock (MockLogger, MockRegistry, MockEmitter)

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_TRAINING_SCRIPT")

# LOẠI BỎ: def load_config(config_path: str) -> Dict[str, Any] vì đã chuyển vào PipelineRunner

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
        logger.info(f"Starting End-to-End Training Workflow for ID: {args.id}")
        
        # 1. Lắp ráp và Khởi tạo Orchestrator Cốt lõi qua Runner
        # Runner sẽ gọi Factory, Factory sẽ tiêm tất cả các Dependencies
        training_orchestrator = PipelineRunner.create_orchestrator(
            config_path=args.config,
            run_id=args.id,
            pipeline_type="training" # <<< YÊU CẦU LOẠI PIPELINE >>>
        )

        # 2. THỰC THI WORKFLOW
        logger.info("Starting Orchestrator run()...")
        final_metrics, endpoint_id = training_orchestrator.run() # Endpoint_id được đổi tên từ model_uri

        # 3. Báo cáo Kết quả
        logger.info("=====================================================")
        logger.info(f"✅ WORKFLOW COMPLETED SUCCESSFULLY. Metrics: {final_metrics}")
        logger.info(f"✅ Deployment Endpoint ID (or Model URI): {endpoint_id}")
        logger.info("=====================================================")

    except FileNotFoundError as e:
        logger.error(f"❌ CONFIG ERROR: {e}")
        exit(1)
    except InvalidConfigError as e:
        logger.error(f"❌ VALIDATION ERROR: Configuration did not pass schema check. Details: {e}")
        exit(1)
    except WorkflowExecutionError as e:
        logger.critical(f"❌ CRITICAL FAILURE: Workflow failed during execution. Details: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"❌ UNEXPECTED ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()