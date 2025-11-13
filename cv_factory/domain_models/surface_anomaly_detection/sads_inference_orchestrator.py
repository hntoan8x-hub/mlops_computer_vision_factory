# domain_models/surface_anomaly_detection/sads_inference_orchestrator.py (FINAL HARDENED VERSION)

import logging
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
import torch
from dataclasses import asdict

# --- Import Abstractions và Contracts (Shared Factory) ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError
from shared_libs.inference.cv_predictor import CVPredictor
from shared_libs.inference.base_cv_predictor import BaseCVPredictor
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
# Giả định utility chung cho xử lý ảnh sau tiền xử lý tồn tại
# from shared_libs.data_processing.cv_utils import crop_and_resize_for_submodel 

# --- Import Schema và Domain Logic ---
from .sads_config_schema import SADSPipelineConfig 
from .sads_postprocessor import SADSPostprocessor
from .sads_output_adapter import SADSOutputAdapter 
from .sads_data_contract import SADSDefect, SADSFinalResult 

logger = logging.getLogger(__name__)

class SADSInferenceOrchestrator(BaseOrchestrator):
    """
    Orchestrator chuyên biệt cho Surface Anomaly Detection System (SADS).
    Quản lý luồng tuần tự 3 mô hình (Detection -> Classification -> Segmentation) 
    và hợp nhất kết quả bằng Domain Adapter.
    """

    def __init__(self, orchestrator_id: str, config: Dict[str, Any], logger_service: BaseTracker, event_emitter: Any):
        
        # 1. Base Init (Validate Config, Inject Logger/Emitter)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 2. Khởi tạo 3 Predictor độc lập
        self.detection_predictor: BaseCVPredictor = self._create_predictor("detection")
        self.classification_predictor: BaseCVPredictor = self._create_predictor("classification")
        self.segmentation_predictor: BaseCVPredictor = self._create_predictor("segmentation")
        
        # 3. Khởi tạo Domain Adapter và Postprocessor (DI Logic)
        postprocessor_params = self.config['domain']['postprocessor']['params']
        self.sads_postprocessor = SADSPostprocessor(**postprocessor_params)
        self.sads_output_adapter = SADSOutputAdapter() 

        self.logger.info("SADS Orchestrator initialized with 3 models and Domain Logic.")

    # --- Phương thức Tiện ích ---

    def _create_predictor(self, module_name: str) -> BaseCVPredictor:
        """
        Tạo và tải mô hình cho một module cụ thể.
        """
        module_config = self.config['models'][module_name]
        
        # SỬ DỤNG CVPredictor CHUNG VÀ TIÊM DUMMY POSTPROCESSOR
        predictor = CVPredictor(
            predictor_id=f"{self.orchestrator_id}-{module_name}",
            config=self.config, 
            postprocessor=object() # Dummy, vì SADS Postprocessor là Decision Engine cuối cùng
        )
        
        # Load mô hình ngay lập tức
        model_uri = module_config['uri']
        predictor.load_model(model_uri)
        
        return predictor
        
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validates the configuration using the Pydantic schema SADSPipelineConfig.
        """
        try:
            SADSPipelineConfig(**config) 
            self.logger.info("Configuration validated successfully against SADSPipelineConfig schema.")
        except Exception as e:
            self.logger.error(f"Invalid SADS configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"SADS config validation failed: {e}") from e

    def _crop_and_resize(self, processed_image: Any, bbox_normalized: List[float]) -> Any:
        """
        Điều phối logic cắt (crop) vùng BBox từ ảnh đã xử lý và resize lại 
        để phù hợp với input của mô hình Classification/Segmentation.
        """
        # NOTE: Logic thực tế nên gọi utility chung (shared_libs/data_processing/cv_utils)
        self.logger.debug(f"Simulating crop and resize for BBox: {bbox_normalized}")
        # Giả định trả về ảnh đã xử lý/cắt/resize
        return processed_image 

    # --- Core Execution Flow (CLEAN INFERENCE FLOW) ---

    def run(self, raw_input: Any) -> Dict[str, Any]:
        """
        Thực hiện luồng Inference tuần tự: Det -> Cls -> Seg -> Adapter -> Postprocessor.
        """
        self.logger.info(f"[{self.orchestrator_id}] Starting SADS sequential inference.")
        
        # 1. PREPROCESS (Chỉ một lần)
        model_input_det = self.detection_predictor.preprocess(raw_input)
        processed_image = model_input_det 
        image_size = processed_image.shape[2:] 
        
        batch_assembly_data: List[Tuple[Dict, Dict, np.ndarray]] = []

        # 2. DETECTION & Standardization
        raw_output_det = self.detection_predictor.predict(model_input_det)
        standardized_detections: List[Dict] = self.detection_predictor.output_adapter.adapt(
            raw_output_det, 
            image_size=image_size
        )
        
        # --- Lặp qua từng ứng viên lỗi (candidate) ---
        for defect in standardized_detections:
            
            bbox_normalized = defect['box']
            
            # 3. CROP & RESIZE
            cropped_input = self._crop_and_resize(processed_image, bbox_normalized)
            
            # 4. CLASSIFICATION & Standardization
            raw_output_cls = self.classification_predictor.predict(cropped_input)
            standardized_classification = self.classification_predictor.output_adapter.adapt(raw_output_cls)
            class_result = standardized_classification[0] 
            
            # 5. SEGMENTATION & Standardization
            raw_output_seg = self.segmentation_predictor.predict(cropped_input)
            standardized_segmentation = self.segmentation_predictor.output_adapter.adapt(raw_output_seg)
            mask_array = standardized_segmentation
            
            # 6. Chuẩn bị cho Hợp nhất Batch
            batch_assembly_data.append((defect, class_result, mask_array))

        # --- Giai đoạn 3: DOMAIN OUTPUT ASSEMBLY ---
        self.logger.info("Executing SADS Domain Output Assembly: 3 Outputs -> SADSDefect Entities.")
        # CHUYỂN LOGIC HỢP NHẤT TỪ ORCHESTRATOR SANG ADAPTER
        unified_defects: List[SADSDefect] = self.sads_output_adapter.assemble_batch(batch_assembly_data)

        # --- Giai đoạn 4: FINAL DECISION ---
        
        # 7. DECISION ENGINE: Gọi hàm decide() nhận Entity SADSDefect
        self.logger.info("Executing SADS Decision Engine (Postprocessor).")
        final_result: SADSFinalResult = self.sads_postprocessor.decide(unified_defects)
        
        # 8. GHI LOG VÀ TRẢ VỀ
        final_result_dict = asdict(final_result) 
        
        self.mlops_tracker.log_metrics({"inference_count": 1, "decision": final_result.final_decision})
        self.emit_event(event_name="sads_inference_complete", payload=final_result_dict)
        
        self.logger.info(f"SADS Pipeline completed. Decision: {final_result.final_decision}.")
        
        return final_result_dict