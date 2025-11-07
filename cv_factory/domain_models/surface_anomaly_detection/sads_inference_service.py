# domain_models/surface_anomaly_detection/sads_inference_service.py

import logging
from typing import Dict, Any, Union, Optional
import numpy as np

# Import Pipeline đã xây dựng
from .sads_pipeline import SADSPipeline 

logger = logging.getLogger(__name__)

class SADSInferenceService:
    """
    Lớp Service đóng vai trò là Endpoint Façade cho SADS.
    Quản lý khởi tạo và luồng I/O của Inference Pipeline.
    """
    
    # Instance cache (sử dụng Singleton pattern đơn giản)
    _pipeline_instance: Optional[SADSPipeline] = None

    def __init__(self, pipeline_config: Dict[str, Any]):
        """
        Khởi tạo Service. Nó sẽ khởi tạo Pipeline và lưu vào cache nếu chưa có.
        """
        self.pipeline_config = pipeline_config
        self.logger = logger
        
        # 1. Khởi tạo Pipeline chỉ một lần duy nhất (Lifecycle Management)
        if SADSInferenceService._pipeline_instance is None:
            self.logger.info("Initializing SADS Pipeline instance (Caching)...")
            try:
                # Sử dụng SADSPipeline làm Façade
                SADSInferenceService._pipeline_instance = SADSPipeline(
                    raw_config=pipeline_config, 
                    orchestrator_id="SADS_Inference_Service_Instance"
                )
            except Exception as e:
                self.logger.critical(f"FATAL: Could not initialize SADS Pipeline: {e}")
                raise RuntimeError("Service failed to initialize underlying pipeline.") from e
        
        self.pipeline = SADSInferenceService._pipeline_instance

    def predict(self, raw_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Giao diện Endpoint chính: Chuyển đổi input thô và chạy Inference Pipeline.
        
        Args:
            raw_input (str | Dict): Dữ liệu thô từ HTTP (ví dụ: Base64 string hoặc JSON payload).

        Returns:
            Dict[str, Any]: Kết quả Decision JSON.
        """
        
        # 1. Adapter Giao thức: Chuyển đổi input thô
        # Predictor.preprocess sẽ xử lý Base64, nhưng ta cần wrapper nhẹ ở đây
        if isinstance(raw_input, str):
            # Giả định input là Base64 string, đóng gói vào Dict
            input_data_for_pipeline = {"image_data": raw_input}
        elif isinstance(raw_input, Dict) and raw_input.get('image_base64'):
             # Input là JSON payload, lấy Base64 string
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