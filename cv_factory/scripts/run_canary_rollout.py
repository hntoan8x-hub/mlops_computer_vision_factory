# scripts/run_canary_rollout.py (HARDENED)

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
logger = logging.getLogger("GLOBAL_CANARY_SCRIPT")

# LOẠI BỎ: def load_config()

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
    parser.add_argument("--name", type=str, required=True, 
                        help="Tên mô hình (ví dụ: defect_detector).")
    parser.add_argument("--id", type=str, default="cv_canary_rollout_01", 
                        help="ID duy nhất cho lần chạy Orchestrator này.")
    args = parser.parse_args()

    try:
        logger.info(f"Starting End-to-End Deployment Workflow for ID: {args.id}")
        
        # 1. Lắp ráp và Khởi tạo Deployment Orchestrator qua Runner
        deployment_orchestrator = PipelineRunner.create_orchestrator(
            config_path=args.config,
            run_id=args.id,
            pipeline_type="deployment" # <<< YÊU CẦU LOẠI PIPELINE >>>
        )

        # 2. THỰC THI WORKFLOW: CANARY DEPLOYMENT
        logger.info(f"Starting CANARY Rollout for {args.uri}. Initial traffic: {args.canary_percent}%.")
        
        endpoint_id = deployment_orchestrator.run(
            model_artifact_uri=args.uri,
            model_name=args.name, # Tên model/endpoint
            mode="canary", # <<< CHẾ ĐỘ TRIỂN KHAI >>>
            new_version_tag=args.uri.split('/')[-1], # Lấy tên version từ URI 
            stable_version=args.stable_version,
            canary_traffic_percent=args.canary_percent 
        )

        # 3. Báo cáo Kết quả
        logger.info("=====================================================")
        logger.info("✅ CANARY ROLLOUT COMPLETED SUCCESSFULLY.")
        logger.info(f"✅ Endpoint ID: {endpoint_id}. Traffic is now split at {args.canary_percent}%.")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Canary rollout failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()