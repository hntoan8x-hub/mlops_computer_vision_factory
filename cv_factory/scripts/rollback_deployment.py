# scripts/rollback_deployment.py (HARDENED)

import argparse
import logging
import os
import json
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError

# <<< NEW: SỬ DỤNG PIPELINE RUNNER >>>
from shared_libs.orchestrators.pipeline_runner import PipelineRunner 


# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_ROLLBACK_SCRIPT")


def main():
    """
    Hàm chính thực thi luồng Rollback khẩn cấp.
    """
    parser = argparse.ArgumentParser(description="Kích hoạt Rollback về phiên bản ổn định (Stable Version).")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML của Deployment.")
    parser.add_argument("--target-version", type=str, required=True, 
                        help="Phiên bản ổn định cần Rollback về (ví dụ: v2.1.0 hoặc 'stable').")
    parser.add_argument("--name", type=str, required=True, 
                        help="Tên mô hình/endpoint cần rollback.")
    parser.add_argument("--id", type=str, default="cv_emergency_rollback_01", 
                        help="ID duy nhất cho lần chạy Orchestrator này.")
    args = parser.parse_args()

    try:
        logger.info(f"Starting Emergency Rollback Workflow for ID: {args.id}")
        
        # 1. Lắp ráp và Khởi tạo Deployment Orchestrator qua Runner
        deployment_orchestrator = PipelineRunner.create_orchestrator(
            config_path=args.config,
            run_id=args.id,
            pipeline_type="deployment" # <<< YÊU CẦU LOẠI PIPELINE >>>
        )

        # 2. THỰC THI WORKFLOW: ROLLBACK
        logger.critical(f"🚨 STARTING EMERGENCY ROLLBACK to version: {args.target_version}")
        
        # Deployer sẽ tự động chuyển 100% traffic về phiên bản ổn định
        endpoint_id = deployment_orchestrator.run(
            model_artifact_uri="models:/rollback/placeholder", # URI không quan trọng trong chế độ rollback
            model_name=args.name,
            mode="rollback", # <<< CHẾ ĐỘ TRIỂN KHAI >>>
            target_version=args.target_version # Tham số cho Rollback
        )

        # 3. Báo cáo Kết quả
        logger.info("=====================================================")
        logger.info("✅ ROLLBACK COMPLETED SUCCESSFULLY.")
        logger.info(f"✅ Endpoint ID: {endpoint_id}. Traffic is 100% on {args.target_version}.")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Rollback failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()