# domain_models/surface_anomaly_detection/sads_inference_service.py (HARDENED)

import logging
from typing import Dict, Any, Union, Optional
import numpy as np

# Import Pipeline đã xây dựng
from .sads_pipeline import SADSPipeline 
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker # Cần cho Mock
from shared_libs.ml_core.monitoring.base_event_emitter import BaseEventEmitter # Cần cho Mock

logger = logging.getLogger(__name__)

class SADSInferenceService:
    """
    Lớp Service đóng vai trò là Endpoint Façade cho SADS.
    Service NHẬN Pipeline đã được khởi tạo từ Factory.
    """
    
    # Loại bỏ _pipeline_instance và logic Singleton thủ công

    def __init__(self, pipeline: SADSPipeline): # <<< NHẬN PIEPELINE ĐÃ KHỞI TẠO >>>
        """
        Khởi tạo Service bằng cách tiêm (Inject) SADS Pipeline đã được cấu hình.
        """
        self.logger = logger
        
        if not isinstance(pipeline, SADSPipeline):
             self.logger.critical("FATAL: Injected pipeline is not an instance of SADSPipeline.")
             raise TypeError("SADSInferenceService requires an initialized SADSPipeline instance.")

        # LƯU Ý: Giả định Service được Factory tạo và tiêm Pipeline
        self.pipeline = pipeline

    def predict(self, raw_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Giao diện Endpoint chính: Chuyển đổi input thô và chạy Inference Pipeline.
        """
        
        # 1. Adapter Giao thức: Chuyển đổi input thô
        input_data_for_pipeline = None
        if isinstance(raw_input, str):
            input_data_for_pipeline = {"image_data": raw_input}
        elif isinstance(raw_input, Dict) and raw_input.get('image_base64'):
             input_data_for_pipeline = {"image_data": raw_input['image_base64']}
        else:
            self.logger.error("Input format not recognized. Expected Base64 string or JSON payload.")
            return {"error": "Invalid input format provided to service."}

        # 2. Ủy quyền Logic Nghiệp vụ
        try:
            # Gọi hàm run_inference của Pipeline Façade
            result = self.pipeline.run_inference(input_data_for_pipeline)
            return result
        except Exception as e:
            self.logger.error(f"Inference execution failed during prediction call: {e}")
            return {"error": "Inference Execution Error", "details": str(e)}

# --- LOGIC FACTORY CẤP CAO ĐỂ KHỞI TẠO SERVICE (CHO BÊN NGOÀI SỬ DỤNG) ---
class SADSServiceFactory:
    """Mô phỏng Factory/Composition Root cấp cao để tạo Service."""
    @staticmethod
    def create_service(full_config: Dict[str, Any]) -> SADSInferenceService:
        
        # MOCKING: Tạo các Service MLOps cần thiết cho Pipeline
        class MockTracker(BaseTracker):
            def log_metrics(self, metrics, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        class MockEmitter(BaseEventEmitter):
            def emit_event(self, name, payload): pass
        
        mlops_tracker = MockTracker()
        event_emitter = MockEmitter()

        # 1. Tạo Pipeline Façade (Tiêm MLOps Services)
        pipeline_instance = SADSPipeline(
            raw_config=full_config, 
            orchestrator_id="SADS_SERVICE_PIPELINE",
            logger_service=mlops_tracker,
            event_emitter=event_emitter
        )
        
        # 2. Tiêm Pipeline vào Service
        return SADSInferenceService(pipeline=pipeline_instance)