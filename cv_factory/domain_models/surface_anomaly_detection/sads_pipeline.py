# domain_models/surface_anomaly_detection/sads_pipeline.py (UPDATED)

import logging
from typing import Dict, Any, Union, List

# Import Utilities và Contracts
from shared_libs.orchestrators.cv_pipeline_factory import CVPipelineFactory
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import WorkflowExecutionError
from shared_libs.ml_core.configs.orchestrator_config_schema import InferenceOrchestratorConfig # Dùng schema chính
import numpy as np

# IMPORT SADS INFERENCE ORCHESTRATOR MỚI
from .sads_inference_orchestrator import SADSInferenceOrchestrator

logger = logging.getLogger(__name__)

class SADSPipeline:
    """
    Façade (mặt tiền) cho Surface Anomaly Detection System (SADS).
    Sử dụng SADSInferenceOrchestrator chuyên biệt để điều phối luồng tuần tự.
    """
    
    def __init__(self, raw_config: Dict[str, Any], orchestrator_id: str = "SADS_Inference_Sequential"):
        """
        Khởi tạo pipeline bằng cách sử dụng CVPipelineFactory, nhưng ép buộc 
        sử dụng SADSInferenceOrchestrator.
        """
        self.orchestrator_id = orchestrator_id
        self.config = raw_config
        
        # 1. Chuẩn bị các service cần thiết (giống logic CVPipelineFactory._create_mlops_services)
        # NOTE: Trong môi trường thực, các service này thường được Dependency Injection từ bên ngoài
        self.mlops_services = self._create_mlops_services_mock() 

        # 2. Khởi tạo Orchestrator chuyên biệt (Thay thế logic CVPipelineFactory.create)
        try:
            self.orchestrator: SADSInferenceOrchestrator = SADSInferenceOrchestrator(
                orchestrator_id=orchestrator_id,
                config=self._merge_config_with_domain_details(raw_config),
                logger_service=self.mlops_services['logger_service'],
                event_emitter=self.mlops_services['event_emitter']
            )
            self.logger.info(f"SADS Pipeline initialized via SADSInferenceOrchestrator.")
        except Exception as e:
             self.logger.error(f"Failed to initialize SADSInferenceOrchestrator: {e}")
             raise RuntimeError(f"SADS Pipeline initialization failed: {e}")


    def _create_mlops_services_mock(self) -> Dict[str, Any]:
        """Mô phỏng việc tạo các service (thực tế được làm bởi CVPipelineFactory)."""
        # Giả định các lớp Mock tồn tại trong môi trường test/mô phỏng
        class MockTracker:
            def log_metrics(self, metrics, **kwargs): self.log_info(f"MOCK LOG: {metrics}")
            def start_run(self, run_name): return self
            def log_info(self, msg): print(msg)
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        
        class MockEmitter:
            def emit_event(self, name, payload): print(f"MOCK EVENT: {name}")

        return {
            "logger_service": MockTracker(),
            "event_emitter": MockEmitter(),
        }
        
    def _merge_config_with_domain_details(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đảm bảo config có các key cần thiết cho Orchestrator (ví dụ: device).
        """
        # Thêm các key cần thiết cho SADSOrchestrator (ví dụ: device, models list)
        merged_config = raw_config.copy()
        
        # Mô phỏng việc Orchestrator cần danh sách 3 mô hình riêng biệt
        merged_config['models'] = {
            "detection": merged_config['model'].copy(), # Giả định model config ban đầu là Detection
            "classification": merged_config.get('classification_model', {"uri": "models:/cls/latest"}),
            "segmentation": merged_config.get('segmentation_model', {"uri": "models:/seg/latest"}),
        }
        merged_config['device'] = 'cuda' # Thêm device config
        
        return merged_config


    def run_inference(self, raw_input: Any) -> Dict[str, Any]:
        """
        Chạy luồng Sequential Inference.
        """
        if not isinstance(self.orchestrator, SADSInferenceOrchestrator):
            raise RuntimeError("Orchestrator type mismatch. Expected SADSInferenceOrchestrator.")
            
        try:
            # Uỷ quyền toàn bộ việc xử lý cho Orchestrator chuyên biệt
            return self.orchestrator.run(raw_input)
        except WorkflowExecutionError as e:
            self.logger.error(f"SADS Inference failed during execution: {e}")
            raise

# --- Khối Cấu hình Test (Cần được bổ sung trong file config thực) ---
SADS_CONFIG_TEST = {
    "pipeline": {"type": "inference"},
    "device": "cuda",
    "model": {"uri": "models:/det/latest", "name": "defect_detector"}, # Model chính
    "classification_model": {"uri": "models:/cls/latest", "name": "defect_classifier"},
    "segmentation_model": {"uri": "models:/seg/latest", "name": "defect_segmenter"},
    "data_ingestion": {"connector_config": {"type": "image"}},
    "preprocessing": {"steps": [{"name": "resize", "params": {"size": 640}}]},
    "domain": {
        "postprocessor": {
            "module_path": "domain_models.surface_anomaly_detection.sads_postprocessor",
            "class_name": "SADSPostprocessor",
            "params": {"defect_confidence_threshold": 0.75, "nms_iou_threshold": 0.3, "max_allowed_defects": 1}
        }
    }
}