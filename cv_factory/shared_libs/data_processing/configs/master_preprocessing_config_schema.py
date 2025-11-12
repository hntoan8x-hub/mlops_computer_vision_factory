import logging
from pydantic import Field, BaseModel
from typing import Optional

# --- Import All Child Schemas ---
# Các schema này đã được tách ra thành file riêng biệt.
from .base_component_schema import BaseConfig
from .image_config_schema import CleaningConfig, AugmentationConfig, FeatureExtractionConfig
from .video_config_schema import VideoProcessingConfig
from .depth_config_schema import DepthProcessingConfig
from .mask_config_schema import MaskProcessingConfig
from .pointcloud_config_schema import PointcloudProcessingConfig
from .text_config_schema import TextProcessingConfig

logger = logging.getLogger(__name__)

class ProcessingConfig(BaseConfig):
    """
    Master schema cho toàn bộ Data Processing và Feature Engineering pipeline.
    
    File này đóng vai trò là Façade (mặt tiền), tổng hợp cấu hình của tất cả các luồng xử lý 
    chuyên biệt (Image, Video, Depth, Mask, Pointcloud, Text).
    """
    
    # --- 1. IMAGE PROCESSING CORE (BẮT BUỘC) ---
    
    # Cleaning, Augmentation, Feature Extraction được xem là cốt lõi cho mọi tác vụ CV.
    cleaning: CleaningConfig = Field(..., description="Cấu hình cho pipeline Image Cleaning bắt buộc.")
    augmentation: AugmentationConfig = Field(..., description="Cấu hình cho pipeline Image Augmentation (tùy chọn, thường cho Training).")
    feature_engineering: FeatureExtractionConfig = Field(..., description="Cấu hình cho Feature Generation/Embedding.")
    
    # --- 2. CÁC LUỒNG DỮ LIỆU ĐẶC BIỆT (TÙY CHỌN) ---
    
    video_processing: Optional[VideoProcessingConfig] = Field(
        None,
        description="Cấu hình cho pipeline Video Processing và Frame Sampling."
    )
    
    depth_processing: Optional[DepthProcessingConfig] = Field(
        None,
        description="Cấu hình cho pipeline Depth Map Processing (xử lý cặp RGB/Depth)."
    )
    
    mask_processing: Optional[MaskProcessingConfig] = Field(
        None,
        description="Cấu hình cho pipeline Mask Processing (xử lý cặp RGB/Mask)."
    )
    
    pointcloud_processing: Optional[PointcloudProcessingConfig] = Field(
        None,
        description="Cấu hình cho pipeline Point Cloud Processing (dữ liệu LiDAR/3D)."
    )
    
    text_processing: Optional[TextProcessingConfig] = Field(
        None,
        description="Cấu hình cho pipeline Text Processing (xử lý văn bản cho OCR/VQA)."
    )