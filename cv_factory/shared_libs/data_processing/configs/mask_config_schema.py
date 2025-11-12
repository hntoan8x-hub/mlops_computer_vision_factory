import logging
from pydantic import Field, validator, NonNegativeInt, BaseModel, constr
from typing import List, Dict, Any, Optional, Union, Literal

logger = logging.getLogger(__name__)

# --- Base Schema (Giả định được kế thừa từ một BaseConfig chung) ---
class BaseConfig(BaseModel):
    class Config:
        extra = "forbid" 
    enabled: bool = Field(True, description="Flag để bật/tắt component này.")
    params: Optional[Dict[str, Any]] = Field(None, description="Các tham số cụ thể của component.")

# --- 1. Atomic Component Step Schema (Định nghĩa riêng cho Mask) ---

class MaskComponentStepConfig(BaseConfig):
    """Schema cho một bước xử lý Mask (Loader, Normalizer, Augmenter, Validator)."""
    type: constr(to_lower=True) = Field(..., description="Tên/loại component Mask (ví dụ: 'mask_loader', 'mask_normalizer').")
    
    @validator('type')
    def validate_known_mask_component(cls, v):
        """Xác thực rằng loại component Mask được hỗ trợ."""
        supported_mask_types = [
            'mask_loader', 'mask_normalizer', 'mask_augmenter', 'mask_validator'
        ]
        if v not in supported_mask_types:
            raise ValueError(f"Unknown Mask component type: {v}. Must be one of: {', '.join(supported_mask_types)}")
        return v

    @validator('params')
    def validate_mask_normalizer_params(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Thi hành các quy tắc consistency cho Mask Normalizer."""
        if values.get('type') == 'mask_normalizer':
            if not v or 'output_format' not in v:
                raise ValueError("Mask Normalizer requires 'output_format' (e.g., 'long_label', 'one_hot').")
            
            output_format = v.get('output_format')
            if output_format == 'one_hot' and 'num_classes' not in v:
                raise ValueError("Mask Normalizer in 'one_hot' mode requires 'num_classes' parameter.")
            
        return v
        
# --- 2. Mask Processing Configuration (Schema Chính) ---

class MaskProcessingConfig(BaseConfig):
    """
    Schema cho toàn bộ Mask Processing pipeline. 
    Quản lý luồng xử lý đồng bộ giữa RGB và Mask/Label Map/BBox.
    """
    
    # 1. Policy Mode: Kiểm soát cách các bước được thực thi
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy mode: 'default' sequential execution, hoặc 'conditional_metadata' (adaptive)."
    )
    
    # 2. Các bước xử lý cụ thể
    
    loader: MaskComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho MaskLoader (tải và căn chỉnh RGB/Mask). Loại: 'mask_loader'."
    )
    
    normalizer: MaskComponentStepConfig = Field(
        ...,
        description="Cấu hình BẮT BUỘC cho MaskNormalizer (chuẩn hóa mask, One-hot, BBox scaling). Loại: 'mask_normalizer'."
    )
    
    augmenter: Optional[MaskComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho MaskAugmenter (lật, xoay đồng bộ). Loại: 'mask_augmenter'."
    )
    
    validator: Optional[MaskComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho MaskValidator (kiểm tra phân phối lớp, chất lượng). Loại: 'mask_validator'."
    )

    # 3. Quy tắc Governance cho toàn bộ pipeline
    
    @validator('loader')
    def validate_loader_type(cls, v: MaskComponentStepConfig) -> MaskComponentStepConfig:
        if v.type != 'mask_loader':
            raise ValueError(f"Loader step must be of type 'mask_loader', but got '{v.type}'.")
        return v
    
    @validator('normalizer')
    def validate_normalizer_type(cls, v: MaskComponentStepConfig) -> MaskComponentStepConfig:
        if v.type != 'mask_normalizer':
            raise ValueError(f"Normalizer step must be of type 'mask_normalizer', but got '{v.type}'.")
        return v

    @validator('augmenter', always=True)
    def validate_augmenter_context(cls, v: Optional[MaskComponentStepConfig], values: Dict[str, Any]) -> Optional[MaskComponentStepConfig]:
        """Rule: Nếu Augmenter được bật, phải có ít nhất một tham số augment được định nghĩa."""
        if v and v.enabled and not v.params:
             logging.warning("Mask Augmenter is enabled but 'params' dictionary is empty. No augmentations will be applied.")
        return v
    
    @validator('params')
    def validate_mask_loader_params(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enforces consistency rules for the 'mask_loader' component."""
        # Ví dụ: Kiểm tra format
        if values.get('type') == 'mask_loader':
            if v and v.get('mask_type') not in ['segmentation', 'bbox', 'keypoints']:
                raise ValueError("MaskLoader 'mask_type' must be one of: 'segmentation', 'bbox', or 'keypoints'.")
        return v
