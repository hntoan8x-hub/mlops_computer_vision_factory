# domain_models/surface_anomaly_detection/sads_config_schema.py (UPDATED)

from pydantic import BaseModel, Field, confloat, NonNegativeInt
from typing import Dict, Any, Optional

class SADSPostprocessorParams(BaseModel):
    """
    Tham số nghiệp vụ cho việc quyết định lỗi bề mặt (Postprocessor).
    """
    # Ngưỡng tin cậy tối thiểu cho một lỗi được chấp nhận (Score < Threshold -> ESCALATE)
    defect_confidence_threshold: confloat(ge=0, le=1) = Field(0.70, description="Ngưỡng score tối thiểu để chấp nhận lỗi. Score < ngưỡng sẽ ESCALATE.")
    
    # Ngưỡng diện tích lỗi tối đa cho phép (Business Rule 2)
    max_area_cm2: confloat(ge=0) = Field(0.5, description="Diện tích lỗi tối đa cho phép (tính bằng cm²). Lỗi > ngưỡng là FAIL.")
    
    # Số lượng lỗi tối đa được chấp nhận (Business Rule 3)
    max_allowed_defects: NonNegativeInt = Field(1, description="Số lượng lỗi tối đa được phép trong một Frame (Frame-level Rule).")
    
    # NOTE: NMS logic được chuyển về Detection Predictor, không cần ở Postprocessor nữa.

class SADSPipelineConfig(BaseModel):
    """
    Cấu hình toàn bộ cho Surface Anomaly Detection System.
    """
    domain_type: str = Field("surface_anomaly_detection", const=True, description="Loại domain.")
    
    # Các model URI cần thiết (được sử dụng bởi Orchestrator)
    models: Dict[str, Any] = Field(..., description="Cấu hình cho Detection, Classification, Segmentation models.")
    
    # Cấu hình Postprocessor
    domain: Dict[str, Any] = Field(..., description="Chứa cấu hình cho postprocessor domain.")