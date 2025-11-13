# scripts/find_best_experiment.py (NEW FILE: Best Model Selection)

import argparse
import logging
import os
import json
from typing import Dict, Any, Tuple, Optional

# --- Import Core Components & Contracts ---
from shared_libs.orchestrators.pipeline_runner import PipelineRunner 
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
from shared_libs.orchestrators.utils.orchestrator_exceptions import WorkflowExecutionError

# --- Cấu hình Logging Cơ bản ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GLOBAL_BEST_EXPERIMENT_SCRIPT")

# Tên file cấu hình sẽ được cập nhật với tham số tốt nhất
OPTIMIZED_CONFIG_OUTPUT = "configs/optimized_training_config.json"


def _find_best_run_details(tracker: BaseTracker, experiment_name: str, metric: str, mode: str) -> Optional[Dict[str, Any]]:
    """
    Sử dụng MLOps Tracker để truy vấn và tìm Run có metric tốt nhất.
    """
    logger.info(f"Querying experiment '{experiment_name}' for best run based on metric '{metric}' ({mode}).")
    
    # MOCKING: Giả định BaseTracker có phương thức find_best_run
    # Trong thực tế, bạn sẽ gọi API của MLflow/WandB
    
    best_run_data = tracker.find_best_run(
        experiment_name=experiment_name,
        metric=metric,
        mode=mode
    )
    
    if best_run_data:
        logger.info(f"Found best run: {best_run_data.get('run_id')} with {metric}={best_run_data.get('metric_score'):.4f}")
        return best_run_data
    
    logger.warning("No suitable run found in the specified experiment.")
    return None

def main():
    """
    Thực thi luồng Experimentation: Query MLOps Tracker và lưu cấu hình tối ưu.
    """
    parser = argparse.ArgumentParser(description="Tìm Run thử nghiệm tốt nhất và lưu cấu hình tối ưu.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Đường dẫn đến file cấu hình JSON/YAML (chứa MLflow config).")
    parser.add_argument("--experiment-name", type=str, required=True, 
                        help="Tên Experiment cần tìm kiếm trong MLOps Platform.")
    parser.add_argument("--optimization-metric", type=str, default="validation_map", 
                        help="Metric cần tối ưu (ví dụ: validation_map, validation_loss).")
    parser.add_argument("--optimization-mode", type=str, default="max", choices=['min', 'max'],
                        help="Chế độ tối ưu ('min' cho loss, 'max' cho accuracy/mAP).")
    args = parser.parse_args()

    try:
        # 1. Khởi tạo MLOps Services (Tracker/Registry)
        tracker_service, registry_service = PipelineRunner.create_mlops_services(args.config)
        
        # 2. Tìm Run tốt nhất
        best_run = _find_best_run_details(
            tracker=tracker_service,
            experiment_name=args.experiment_name,
            metric=args.optimization_metric,
            mode=args.optimization_mode
        )

        if not best_run:
            logger.critical("❌ FAILED: Could not find the best experiment run. Aborting.")
            exit(1)
        
        # 3. Lấy Params và Artifacts
        # Giả định: Các tham số training (batch_size, epochs, learning_rate) và 
        # Cấu hình đầy đủ (Full Config) được lưu dưới dạng Params hoặc Artifacts trong Best Run
        
        best_params = best_run.get('params', {})
        full_config_artifact_path = best_run.get('full_config_path', 'config.json') 
        
        # MOCKING: Tải cấu hình đầy đủ từ Artifact Store
        # current_config = tracker_service.download_artifact(best_run['run_id'], full_config_artifact_path)
        current_config = PipelineRunner.load_config(args.config) # Sử dụng config hiện tại làm base
        
        # 4. Cập nhật và lưu cấu hình tối ưu
        # Cập nhật các tham số tìm được vào TrainingConfig
        if best_params.get('trainer_batch_size'):
             current_config['trainer']['batch_size'] = int(best_params['trainer_batch_size'])
        
        current_config['pipeline']['type'] = 'training' # Đảm bảo là training pipeline
        current_config['metadata'] = {
            "source_experiment": args.experiment_name,
            "source_run_id": best_run.get('run_id'),
            "optimized_metric": best_run.get('metric_score')
        }

        with open(OPTIMIZED_CONFIG_OUTPUT, 'w') as f:
            json.dump(current_config, f, indent=4)
        
        logger.info("=====================================================")
        logger.info("✅ BEST EXPERIMENT FOUND & OPTIMIZED CONFIG SAVED.")
        logger.info(f"  Configuration saved to: {OPTIMIZED_CONFIG_OUTPUT}")
        logger.info("=====================================================")

    except WorkflowExecutionError as e:
        logger.critical(f"❌ CRITICAL FAILURE: Workflow failed. Details: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"❌ UNEXPECTED ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()