# domain_models/surface_anomaly_detection/sads_inference_orchestrator.py

import logging
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
import torch

# Import Abstractions và Contracts cần thiết
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError
from shared_libs.inference.cv_predictor import CVPredictor
from shared_libs.inference.base_cv_predictor import BaseCVPredictor, PredictionOutput
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
from shared_libs.ml_core.output_adapter.output_adapter_factory import OutputAdapterFactory
from shared_libs.ml_core.output_adapter.base_output_adapter import BaseOutputAdapter

# Import Schema nghiệp vụ
from .sads_config_schema import SADSPipelineConfig 
from .sads_postprocessor import SADSPostprocessor # Logic nghiệp vụ (Decision Engine)

logger = logging.getLogger(__name__)

class SADSInferenceOrchestrator(BaseOrchestrator):
    """
    Orchestrator chuyên biệt cho Surface Anomaly Detection System (SADS).
    Quản lý luồng tuần tự 3 mô hình: Detection -> Classification -> Segmentation.
    """

    def __init__(self, orchestrator_id: str, config: Dict[str, Any], logger_service: BaseTracker, event_emitter: Any):
        
        # 1. Base Init (Validate Config, Inject Logger/Emitter)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.device = torch.device(self.config['device'] if 'device' in self.config else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Khởi tạo 3 Predictor độc lập
        self.detection_predictor: BaseCVPredictor = self._create_predictor("detection")
        self.classification_predictor: BaseCVPredictor = self._create_predictor("classification")
        self.segmentation_predictor: BaseCVPredictor = self._create_predictor("segmentation")
        
        # 3. Khởi tạo Domain Postprocessor (Decision Engine)
        postprocessor_params = self.config['domain']['postprocessor']['params']
        self.sads_postprocessor = SADSPostprocessor(**postprocessor_params)

    # --- Phương thức Tiện ích ---

    def _create_predictor(self, module_name: str) -> BaseCVPredictor:
        """
        Tạo và tải mô hình cho một module cụ thể (Detection, Classification, Segmentation).
        """
        module_config = self.config['models'][module_name]
        
        # Cấu hình Postprocessor động cho mỗi Predictor (chỉ làm Framework Cleanup)
        # SADS Postprocessor sẽ được gọi sau cùng, bên ngoài Predictor
        dummy_postprocessor = object() 
        
        predictor = CVPredictor(
            predictor_id=f"{self.orchestrator_id}-{module_name}",
            config=self.config, # Truyền config tổng thể
            postprocessor=dummy_postprocessor # Inject dummy, logic sẽ được xử lý ở đây
        )
        
        # Load mô hình ngay lập tức
        model_uri = module_config['uri']
        predictor.load_model(model_uri)
        
        return predictor
        
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        [IMPLEMENTED] Validates the configuration using the Pydantic schema SADSPipelineConfig.
        """
        try:
            # Sử dụng SADSPipelineConfig cho validation
            SADSPipelineConfig(**config) 
            self.logger.info("Configuration validated successfully against SADSPipelineConfig schema.")
        except Exception as e:
            self.logger.error(f"Invalid SADS configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"SADS config validation failed: {e}") from e

    # --- Core Execution Flow ---

    def run(self, raw_input: Any) -> Dict[str, Any]:
        """
        [IMPLEMENTED] Thực hiện luồng Inference tuần tự (Detection -> Classification -> Segmentation).
        """
        self.logger.info(f"[{self.orchestrator_id}] Starting SADS sequential inference.")
        
        # 1. PREPROCESS (Chỉ cần chạy Preprocess 1 lần duy nhất cho Prediction 1)
        # Giả định Preprocess Pipeline giống nhau cho cả 3 mô hình (resize, normalize)
        model_input_det = self.detection_predictor.preprocess(raw_input)
        
        # Dữ liệu ảnh sau tiền xử lý (cần cho các mô hình sau và Explainer)
        processed_image = model_input_det 
        
        # Khởi tạo kết quả chính
        combined_defects: List[Dict[str, Any]] = []

        # --- Giai đoạn 1: DETECTION ---
        
        # 2. DETECT
        raw_output_det = self.detection_predictor.predict(model_input_det)
        
        # 3. ADAPT & CLEANUP (Chuyển Tensor -> NumPy List[Dict])
        # SỬ DỤNG Predictor Postprocessor và Output Adapter của Detection Predictor
        intermediate_output_det = self.detection_predictor.postprocess(raw_output_det)
        standardized_detections = self.detection_predictor.output_adapter.adapt(
            intermediate_output_det, 
            image_size=processed_image.shape[2:] 
        )
        
        # 4. CHỌN LỖI (Sau NMS, v.v.)
        # Logic NMS/Lọc ban đầu được thực hiện ở đây (có thể dùng SADS Postprocessor NMS)
        # Giả sử chúng ta có List[Dict] BBox/Score/Class
        
        # --- Lặp qua từng lỗi được phát hiện ---
        for defect in standardized_detections:
            
            bbox_normalized = defect['box']
            # Cần cắt vùng BBox từ ảnh thô/đã xử lý cho các mô hình tiếp theo
            
            # --- Giai đoạn 2: CLASSIFICATION (trên vùng lỗi) ---
            
            # 5. CROP & RESIZE (Lấy vùng BBox)
            # Giả định có hàm crop/resize từ shared_libs/data_processing
            cropped_input = self._crop_and_resize(processed_image, bbox_normalized)
            
            # 6. CLASSIFY
            raw_output_cls = self.classification_predictor.predict(cropped_input)
            intermediate_output_cls = self.classification_predictor.postprocess(raw_output_cls)
            standardized_classification = self.classification_predictor.output_adapter.adapt(intermediate_output_cls)
            
            # Lấy kết quả Classification (ví dụ: class_id và confidence)
            class_result = standardized_classification[0] 
            
            # --- Giai đoạn 3: SEGMENTATION (trên vùng lỗi) ---
            
            # 7. SEGMENT
            raw_output_seg = self.segmentation_predictor.predict(cropped_input)
            intermediate_output_seg = self.segmentation_predictor.postprocess(raw_output_seg)
            standardized_segmentation = self.segmentation_predictor.output_adapter.adapt(intermediate_output_seg)
            
            # Lấy kết quả Mask (ví dụ: mask array [H, W])
            mask_array = standardized_segmentation # Giả định array
            
            # --- HỢP NHẤT TẠM THỜI ---
            combined_defects.append({
                "bbox": bbox_normalized.tolist(),
                "score": defect['score'],
                "class_id": class_result['class_id'],
                "mask_data": mask_array.tolist() # Lưu mask (cần Base64 trong thực tế)
            })

        # --- Giai đoạn 4: FINAL DECISION ---
        
        # 8. POSTPROCESS VÀ DECISION (Sử dụng Domain Postprocessor)
        final_result = self.sads_postprocessor.run(combined_defects, config=self.config['domain'])
        
        # 9. GHI LOG (Sử dụng Tracker/Emitter)
        self.mlops_tracker.log_metrics({"inference_count": 1, "decision": final_result['decision']})
        self.emit_event(event_name="sads_inference_complete", payload=final_result)
        
        self.logger.info(f"SADS Pipeline completed. Decision: {final_result['decision']}.")
        
        return final_result


    def _crop_and_resize(self, processed_image: Any, bbox_normalized: List[float]) -> Any:
        """
        Mô phỏng logic cắt (crop) vùng BBox từ ảnh đã xử lý và resize lại 
        để phù hợp với input của mô hình Classification/Segmentation.
        """
        self.logger.debug("Simulating crop and resize for sub-task model input.")
        return processed_image # Trả về ảnh gốc để mô phỏng tính nhất quán