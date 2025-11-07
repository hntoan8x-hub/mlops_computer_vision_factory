# shared_libs/ml_core/output_adapter/configs/output_adapter_config_schema.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal, Union, List
import numpy as np

# Định nghĩa các loại tác vụ CV được hỗ trợ
TaskType = Literal[
    "classification", 
    "detection", 
    "segmentation", 
    "ocr", 
    "embedding", 
    "keypoint"
]

# --- 1. Sub-Schemas: Cấu hình chuyên biệt theo Task ---

class ClassificationAdapterParams(BaseModel):
    """Cấu hình tham số cho Classification Adapter."""
    # Xác định xem đầu ra có phải là logits (cần softmax) hay không
    is_logits: bool = Field(True, description="Nếu True, đầu ra mô hình là logits và cần áp dụng softmax.")
    # Ngưỡng tin cậy (chỉ dùng cho mục đích lọc/báo cáo)
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Ngưỡng tin cậy tối thiểu để trả về dự đoán.")

class DetectionAdapterParams(BaseModel):
    """Cấu hình tham số cho Detection Adapter."""
    # Định dạng BBox đầu ra mong muốn (chuẩn hóa về 0-1 hay pixel)
    normalize: bool = Field(True, description="Nếu True, chuẩn hóa BBox về [0, 1].")
    # Định dạng tọa độ BBox đầu vào (ví dụ: 'xyxy' hoặc 'xywh')
    input_bbox_format: Literal["xyxy", "xywh"] = Field("xyxy", description="Định dạng tọa độ BBox thô đầu vào.")
    # Tên key chứa BBoxes trong Dict đầu ra của mô hình (ví dụ: FasterRCNN)
    box_key: str = Field("boxes", description="Tên key chứa tensor BBox trong đầu ra thô.")

class SegmentationAdapterParams(BaseModel):
    """Cấu hình tham số cho Segmentation Adapter."""
    # Chỉ số lớp cần bỏ qua (thường là background)
    ignore_index: Optional[int] = Field(None, description="Chỉ số lớp cần bỏ qua khi chuẩn hóa (ví dụ: background).")
    # Có áp dụng argmax lên Logits 4D ([B, C, H, W]) không
    is_logits: bool = Field(True, description="Nếu True, đầu ra là logits và cần áp dụng argmax.")

class EmbeddingAdapterParams(BaseModel):
    """Cấu hình tham số cho Embedding Adapter."""
    # Kích thước vector dự kiến (để kiểm tra)
    embedding_dim: int = Field(512, gt=0, description="Kích thước vector đặc trưng dự kiến.")
    # Có chuẩn hóa vector về độ dài đơn vị (L2 norm) không
    normalize_vector: bool = Field(True, description="Nếu True, áp dụng L2 normalization lên feature vector.")

# --- 2. Main Config Schema: Cấu trúc chung cho Output Adapter ---

class OutputAdapterConfig(BaseModel):
    """
    Schema cấu hình chính cho Output Adapter.
    """
    
    # [REQUIRED] Loại tác vụ CV (dùng cho OutputAdapterFactory)
    task_type: TaskType = Field(
        ..., description="Loại tác vụ CV mà Adapter này phục vụ."
    )
    
    # [CONFIG CHUNG] Có nên trả về kết quả dưới dạng Dict không
    return_dict: bool = Field(False, description="Nếu True, Adapter trả về Dict[key, np.ndarray], nếu không, trả về np.ndarray hoặc List[Dict].")
    
    # [CONFIG CHI TIẾT THEO TASK]
    # Trường này sẽ chứa Sub-Schemas tương ứng với task_type
    params: Union[
        ClassificationAdapterParams,
        DetectionAdapterParams,
        SegmentationAdapterParams,
        EmbeddingAdapterParams,
        Dict[str, Any] # Fallback cho các params chưa định nghĩa
    ] = Field({}, description="Cấu hình tham số chi tiết cho từng loại Adapter.")