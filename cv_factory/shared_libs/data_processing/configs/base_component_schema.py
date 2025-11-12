import logging
from pydantic import Field, validator, NonNegativeInt, conlist, BaseModel, constr
from typing import List, Dict, Any, Union, Literal, Optional

logger = logging.getLogger(__name__)

# NOTE: Giả định BaseConfig đã được Hardening với extra="forbid"
class BaseConfig(BaseModel):
    """Base class for all configuration schemas to enforce general rules."""
    class Config:
        # Sử dụng 'forbid' trong Production để tăng tính an toàn cấu hình
        extra = "forbid" 
    enabled: bool = Field(True, description="Flag để bật/tắt component này.")
    params: Optional[Dict[str, Any]] = Field(None, description="Các tham số cụ thể của component.")

# --- Atomic Component Step Schema (Universal) ---

class ComponentStepConfig(BaseConfig):
    """
    Schema chung cho một bước bất kỳ trong pipeline.
    
    'type' phải tương ứng với key trong ComponentFactory của từng luồng (Image, Depth, etc.).
    """
    type: constr(to_lower=True) = Field(..., description="The type/name of the component (e.g., 'resizer', 'text_tokenizer').")
    
    @validator('type')
    def validate_known_component(cls, v):
        """Validates that the component type is supported across the entire factory system."""
        # Danh sách này cần được mở rộng liên tục khi các component mới được thêm vào.
        supported = [
            # Image Core
            'resizer', 'normalizer', 'color_space', 
            # Augmentation
            'flip_rotate', 'noise_injection', 'cutmix', 'mixup',
            # Feature/Embedding
            'dim_reducer', 'cnn_embedder', 'vit_embedder',
            'hog_extractor', 'sift_extractor', 'orb_extractor',
            # Depth
            'loader', 'normalizer', 'augmenter', 'validator', 'converter',
            # Mask
            'mask_loader', 'mask_normalizer', 'mask_augmenter', 'mask_validator',
            # Video
            'video_resizer', 'frame_rate_adjuster', 'video_noise_reducer', 'motion_stabilizer',
            'uniform_sampler', 'keyframe_extractor', 'motion_aware_sampler', 'policy_sampler',
            # Point Cloud
            'pointcloud_loader', 'pointcloud_voxelizer', 'pointcloud_normalizer', 'pointcloud_augmenter',
            # Text
            'text_loader', 'text_tokenizer', 'text_augmenter', 'text_validator'
        ]
        if v not in supported:
            logger.warning(f"Unknown component type '{v}'. Please ensure it is correctly registered in the Component Factory.")
            # WARNING thay vì raise error để cho phép factory con kiểm tra.
            
        return v
    
    @validator('params')
    def validate_normalizer_params(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enforces consistency rules specifically for the 'normalizer' component."""
        if values.get('type') == 'normalizer':
            if not v or 'mean' not in v or 'std' not in v:
                raise ValueError("Normalizer component requires 'mean' and 'std' parameters to be defined.")
            
            mean = v.get('mean')
            std = v.get('std')
            
            if type(mean) is not type(std):
                 raise ValueError("Normalizer: 'mean' and 'std' must be of the same type (e.g., both list or both float).")

            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
                if len(mean) != len(std):
                    raise ValueError(f"Normalizer: 'mean' ({len(mean)} elements) and 'std' ({len(std)} elements) must have the same length.")

        return v