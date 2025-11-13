# scripts/run_retrain_check.py (NEW FILE)

import argparse
import logging
import time
import os
import json
from typing import Dict, Any, Tuple

# --- Import các thành phần Cốt lõi ---
from shared_libs.orchestrators.pipeline_runner import PipelineRunner
from shared_libs.orchestrators.utils.orchestrator_exceptions import WorkflowExecutionError
from shared_libs.ml_core.retraining.tmr_facade import TMRFacade # Type Hint

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_RETRAIN_CHECK_SCRIPT")

def main():
    """
    Hàm chính thực thi luồng kiểm tra TMR (Monitoring -> Triggers -> Submit Job).
    """
    parser = argparse.ArgumentParser(description="Kích hoạt luồng kiểm tra Retraining TMR.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML (chứa Monitoring/Retraining config).")
    parser.add_argument("--id", type=str, default=f"tmr_check_{int(time.time())}", 
                        help="ID duy nhất cho lần chạy.")
    args = parser.parse_args()

    try:
        logger.info(f"Starting TMR Check Workflow for ID: {args.id}")
        
        # 1. Lắp ráp và Khởi tạo TMRFacade (Glue) qua Runner
        # Runner sẽ gọi Factory, Factory sẽ tiêm Monitor và Retrain Orchestrators
        tmr_facade: TMRFacade = PipelineRunner.create_orchestrator(
            config_path=args.config,
            run_id=args.id,
            pipeline_type="retrain_check" # <<< YÊU CẦU LOẠI PIPELINE TMR >>>
        )

        # 2. CHUẨN BỊ DATA (MOCK)
        # Trong thực tế, bạn sẽ cần:
        # - Lấy reference_data (embedding/feature phân phối từ training set)
        # - Lấy current_data (embedding/feature/metrics/predictions từ dữ liệu phục vụ gần nhất)
        reference_data = object()
        current_data = object() 
        logger.warning("Using mock data. Implement data collection logic here.")
        
        # 3. THỰC THI WORKFLOW
        is_triggered, reasons = tmr_facade.run_tmr_workflow(
            reference_data=reference_data,
            current_data=current_data
        )

        # 4. Báo cáo Kết quả
        logger.info("=====================================================")
        if is_triggered:
            logger.critical(f"🚨 RETRAINING INITIATED! Reasons: {', '.join(reasons)}")
        else:
            logger.info("✅ TMR Check completed. No retraining required.")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: TMR Check script failed. Details: {e}")
        exit(1)

if __name__ == "__main__":
    main()