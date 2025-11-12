import logging
from pydantic import Field, validator
from typing import Dict, Any, Optional, Literal

# Import Base Schemas
from .base_component_schema import BaseConfig, ComponentStepConfig

logger = logging.getLogger(__name__)

class DepthProcessingConfig(BaseConfig):
    """
    Schema cho toàn bộ Depth Processing pipeline. 
    Quản lý luồng xử lý đồng bộ giữa RGB và Depth Map.
    """
    
    policy_mode: Literal["default"] = Field(
        "default",
        description="Policy for depth processing: 'default' sequential execution."
    )
    
    loader: ComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho DepthLoader (tải và căn chỉnh RGB/Depth). Loại: 'loader'."
    )
    
    normalizer: ComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho DepthNormalizer (chuẩn hóa giá trị độ sâu). Loại: 'normalizer'."
    )
    
    augmenter: Optional[ComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho DepthAugmenter (lật, thêm nhiễu). Loại: 'augmenter'."
    )
    
    validator: Optional[ComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho DepthValidator (kiểm tra phạm vi, chất lượng). Loại: 'validator'."
    )
    
    converter: Optional[ComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho DisparityConverter (Depth <-> Disparity). Loại: 'converter'."
    )
    
    @validator('loader', 'normalizer', 'augmenter', 'validator', 'converter', always=True)
    def validate_depth_component_types(cls, v: Optional[ComponentStepConfig], values: Dict[str, Any], field: Any) -> Optional[ComponentStepConfig]:
        """Rule: Đảm bảo các component Depth có type đúng."""
        if v is None or not v.enabled:
            return v
        
        # Tên loại mong muốn là key của trường (ví dụ: 'loader', 'normalizer')
        expected_type = field.name
        if v.type.lower() != expected_type:
            raise ValueError(
                f"DepthProcessingConfig field '{field.name}' requires component type '{expected_type}', but got '{v.type}'."
            )
        return v