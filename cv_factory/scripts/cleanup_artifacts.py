# scripts/cleanup_artifacts.py

import argparse
import logging
import os
import json
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
from shared_libs.ml_core.mlflow_service.implementations.mlflow_registry import MLflowRegistry as MockRegistry # Registry Service
from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger as MockLogger # Tracker Service

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_CLEANUP_SCRIPT")

def main():
    """
    Hàm chính thực thi luồng dọn dẹp tài nguyên (Cleanup Artifacts).
    """
    parser = argparse.ArgumentParser(description="Dọn dẹp các phiên bản mô hình và artifacts cũ.")
    parser.add_argument("--model-name", type=str, required=True, 
                        help="Tên mô hình cần dọn dẹp (ví dụ: defect_detector).")
    parser.add_argument("--keep-latest", type=int, default=5, 
                        help="Số lượng phiên bản mới nhất cần giữ lại trong Registry.")
    parser.add_argument("--cleanup-runs-older-than-days", type=int, default=180, 
                        help="Xóa các run MLflow cũ hơn X ngày.")
    args = parser.parse_args()

    try:
        # 1. Khởi tạo Registry Service (Mock/Real)
        registry_service = MockRegistry()
        logger.info(f"Starting cleanup process for model: {args.model_name}")

        # --- A. Cleanup Model Registry ---
        
        logger.info(f"  1. Cleaning up Model Registry: Keeping {args.keep_latest} latest versions.")
        
        # MÔ PHỎNG: Dùng hàm của Registry Service để xóa/lưu trữ các phiên bản cũ
        # registry_service.archive_old_versions(args.model_name, keep_count=args.keep_latest)
        logger.warning("Simulating: Archiving versions older than version_N-5.")
        
        # --- B. Cleanup MLflow Runs / Artifacts ---
        
        logger.info(f"  2. Cleaning up MLflow Runs older than {args.cleanup_runs_older_than_days} days.")
        tracker_service = MockLogger() # Dùng tracker service để xóa runs
        # tracker_service.delete_expired_runs(days=args.cleanup_runs_older_than_days)
        logger.warning(f"Simulating: Deleting runs older than {args.cleanup_runs_older_than_days} days.")


        logger.info("=====================================================")
        logger.info("✅ CLEANUP COMPLETED SUCCESSFULLY.")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Cleanup script failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()