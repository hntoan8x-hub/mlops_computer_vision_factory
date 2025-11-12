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
    "keypoint",
    "depth_estimation",
    "pointcloud_processing"
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

class DepthAdapterParams(BaseModel):
    """Cấu hình tham số cho Depth Estimation Adapter."""
    # Depth Estimation có thể cần ngưỡng tối thiểu/tối đa để lọc (ví dụ: MiDaS)
    min_depth: float = Field(0.1, ge=0.0, description="Ngưỡng độ sâu tối thiểu (meters) để loại bỏ outlier.")
    max_depth: float = Field(10.0, gt=0.0, description="Ngưỡng độ sâu tối đa (meters).")
    # Có nên squeeze (loại bỏ) chiều kênh đơn (1) không
    squeeze_channel: bool = Field(True, description="Nếu True, squeeze tensor/array từ [B, 1, H, W] về [B, H, W].")

class PointCloudAdapterParams(BaseModel): # <<< SCHEMA MỚI >>>
    """Cấu hình tham số cho Point Cloud Adapter (Detection/Segmentation)."""
    # Key chứa BBox 3D trong đầu ra thô (Detection 3D)
    box_3d_key: Optional[str] = Field(None, description="Tên key chứa BBox 3D (x, y, z, l, w, h, yaw).")
    # Có chuẩn hóa tọa độ (ví dụ: chia cho kích thước voxel map) không
    normalize_coordinates: bool = Field(False, description="Nếu True, chuẩn hóa tọa độ BBox/Points.")
    # Chỉ số kênh chứa lớp dự đoán (cho Segmentation 3D)
    segmentation_channel: Optional[int] = Field(None, description="Chỉ số kênh chứa lớp dự đoán trong tensor đầu ra.")


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
        DepthAdapterParams,
        PointCloudAdapterParams,# <<< UPDATED >>>
        Dict[str, Any] 
    ] = Field({}, description="Cấu hình tham số chi tiết cho từng loại Adapter.")