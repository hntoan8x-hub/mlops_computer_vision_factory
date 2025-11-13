# api_service/dependencies.py

import logging
from typing import Dict, Any, Optional

# --- Import Logic Domain & Factory ---
from domain_models.surface_anomaly_detection.sads_inference_service import SADSInferenceService, SADSServiceFactory 
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker # Contracts for MLOps

logger = logging.getLogger(__name__)

# Cache Service Instance
# Biến này sẽ giữ instance của Service/Pipeline sau khi khởi tạo (Singleton)
_sads_service_instance: Optional[SADSInferenceService] = None
_app_config: Dict[str, Any] = {} # Giả định cấu hình được tải vào đây

def set_app_config(config: Dict[str, Any]):
    """Dùng để tiêm cấu hình toàn bộ ứng dụng."""
    global _app_config
    _app_config = config
    
def initialize_sads_service():
    """
    Hàm Khởi tạo Service Cốt lõi (Được gọi BÊN NGOÀI, thường là trong main.py/lifecycle).
    Đây là nơi Mô hình SADS (3 thành phần) được tải VÀO BỘ NHỚ.
    """
    global _sads_service_instance
    
    if _sads_service_instance is None:
        logger.info("--- FASTAPI LIFECYCLE: Initializing SADS Service & Loading Models ---")
        try:
            # Sử dụng Factory cấp cao nhất để tạo và tiêm các Dependency cần thiết
            # (Factory đã xử lý việc tạo Pipeline, Orchestrator, Predictors, etc.)
            _sads_service_instance = SADSServiceFactory.create_service(_app_config)
            logger.info("SADS Service initialization COMPLETE. Models loaded.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize SADS Service (Check MLflow URI/Model Loading): {e}")
            # Trong production, nên exit hoặc raise Runtime Error nghiêm trọng
            raise RuntimeError(f"Service startup failed: {e}") from e


def get_sads_service() -> SADSInferenceService:
    """
    FastAPI Dependency: Trả về instance Service đã được khởi tạo (Singleton).
    """
    if _sads_service_instance is None:
        logger.error("SADS Service not initialized. Ensure initialize_sads_service() was called.")
        raise RuntimeError("Service is not available.")
    return _sads_service_instance