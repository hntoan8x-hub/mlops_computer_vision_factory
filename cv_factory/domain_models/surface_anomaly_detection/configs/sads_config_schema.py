# domain_models/surface_anomaly_detection/sads_config_schema.py

from pydantic import BaseModel, Field, confloat, NonNegativeInt
from typing import Dict, Any, Optional

class SADSPostprocessorParams(BaseModel):
    """
    Tham số nghiệp vụ cho việc quyết định lỗi bề mặt (Postprocessor).
    """
    # Ngưỡng tin cậy tối thiểu cho một lỗi được chấp nhận (Score > Threshold)
    defect_confidence_threshold: confloat(ge=0, le=1) = Field(0.70, description="Ngưỡng score tối thiểu để chấp nhận một dự đoán là lỗi.")
    
    # Ngưỡng IoU cho Non-Maximum Suppression (NMS)
    nms_iou_threshold: confloat(ge=0, le=1) = Field(0.45, description="Ngưỡng IoU cho Non-Maximum Suppression (loại bỏ hộp trùng lặp).")
    
    # Ngưỡng diện tích tối thiểu (Tính bằng pixel hoặc normalized)
    min_defect_area_normalized: confloat(ge=0, le=1) = Field(0.001, description="Diện tích lỗi tối thiểu (normalized [0,1]) để bị coi là FAIL.")
    
    # Số lượng lỗi tối đa được chấp nhận (ví dụ: > 1 lỗi nhỏ vẫn PASS)
    max_allowed_defects: NonNegativeInt = Field(0, description="Số lượng lỗi tối đa được phép (0 = chỉ cần 1 lỗi > threshold là FAIL).")

class SADSPipelineConfig(BaseModel):
    """
    Cấu hình toàn bộ cho Surface Anomaly Detection System.
    """
    domain_type: str = Field("surface_anomaly_detection", const=True, description="Loại domain.")
    
    postprocessor_params: SADSPostprocessorParams = Field(..., description="Tham số nghiệp vụ cho Postprocessor.")
    
    # Cấu hình MLOps (được sử dụng bởi Orchestrator)
    model_uri: str = Field(..., description="URI của mô hình Detection đã đăng ký (MLflow).")
    task_type: str = Field("detection", const=True, description="Tác vụ chính của mô hình.")
    
    # Cấu hình các bước khác (có thể mở rộng)
    detection_model_params: Dict[str, Any] = Field({}, description="Tham số cho mô hình Detection (ví dụ: IoU/Conf model).")