# scripts/check_model_health.py (HARDENED)

import argparse
import logging
import os
import json
from typing import Dict, Any

# --- Import các thành phần Cốt lõi ---
from shared_libs.orchestrators.utils.orchestrator_exceptions import WorkflowExecutionError
from shared_libs.orchestrators.pipeline_runner import PipelineRunner # <<< SỬ DỤNG RUNNER >>>
from shared_libs.monitoring.event_emitter import ConsoleEventEmitter as MockEmitter
# NOTE: Cần thêm logic để tạo DataLoader cho dữ liệu Production

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_HEALTH_CHECK_SCRIPT")

def main():
    """
    Hàm chính thực thi luồng kiểm tra chất lượng mô hình (Model Health Check).
    """
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng mô hình (Health Check) trên dữ liệu Production.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML (cấu hình Evaluator).")
    parser.add_argument("--model-uri", type=str, required=True, 
                        help="URI của mô hình (ví dụ: models:/defect_detector/canary).")
    parser.add_argument("--id", type=str, default="cv_health_check_01", 
                        help="ID duy nhất cho lần chạy.")
    args = parser.parse_args()

    try:
        # 1. Khởi tạo Model và Evaluator qua Runner (Hardening)
        # Runner đảm bảo Predictor được tạo, Model được tải, và Evaluator được lắp ráp.
        predictor, evaluation_orchestrator, raw_config = PipelineRunner.create_model_and_evaluator(
            config_path=args.config,
            model_uri=args.model_uri
        )
        
        # 2. Tải dữ liệu Production (Điểm cần triển khai tiếp theo)
        logger.warning("Cần triển khai logic Data Loader Factory để tạo Production Test Loader.")
        test_loader = object() # Giả định DataLoader
        
        # 3. THỰC THI EVALUATION 
        # Sử dụng Model đã tải của Predictor (predictor.model)
        final_metrics = evaluation_orchestrator.evaluate(
            model=predictor.model, 
            data_loader=test_loader, 
            visualize_explanations=False
        )
        
        # 4. Kiểm tra Quality Gate (Logic giữ nguyên)
        evaluator_config = raw_config.get('evaluator', {})
        min_required_map = evaluator_config.get('min_required_map', 0.75)
        current_map = final_metrics.get('metrics', {}).get('map', 0.0) 
        
        health_status = "PASS" if current_map >= min_required_map else "FAIL"
        
        logger.info("=====================================================")
        logger.info(f"✅ MODEL HEALTH CHECK STATUS: {health_status}")
        logger.info(f"  Current mAP: {current_map:.4f} (Required: {min_required_map})")
        logger.info("=====================================================")

        if health_status == "FAIL":
             MockEmitter().emit_event("model_health_check_fail", {"model_uri": args.model_uri, "metric": "mAP"})
             exit(1)

    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Health Check script failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()