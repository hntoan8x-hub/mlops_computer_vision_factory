# shared_libs/ml_core/pipeline_components_cv/configs/component_config_schema.py

from pydantic import Field, validator, BaseModel, constr
from typing import Dict, Any, Optional

# --- Import tất cả các *Params Schema từ các file nhỏ hơn ---
from .schemas.core_cv_schemas import (
    ResizerParams, NormalizerParams, ColorSpaceParams, FlipRotateParams, NoiseInjectionParams, 
    CutMixParams, MixupParams, DimReducerParams, HOGParams, SIFTParams, ORBParams, 
    CNNEmbedderParams, ViTEmbedderParams
)
from .schemas.depth_schemas import DepthCleanerParams, DepthAugmenterParams
from .schemas.mask_schemas import MaskCleanerParams, MaskAugmenterParams
from .schemas.pointcloud_schemas import PointCloudCleanerParams, PointCloudAugmenterParams
from .schemas.text_schemas import TextTokenizerParams, TextAugmenterParams

# --- Base Configuration ---
class BaseConfig(BaseModel):
    """Base schema for all component configurations."""
    class Config:
        extra = "forbid" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")

# --- Master Step Configuration (The Final Product) ---

class PipelineStepConfig(BaseConfig):
    """
    Master schema for a single step in the MLOps processing pipeline. 
    It delegates parameter validation based on the 'type' field to specific schemas.
    """
    type: constr(to_lower=True) = Field(..., description="The Adapter component type (e.g., 'resizer', 'cnn_embedder').")
    
    # PARAM_MAP được định nghĩa ngay trong lớp Master này
    _PARAM_MAP = {
        # CORE CV (13)
        "resizer": ResizerParams, "normalizer": NormalizerParams, "color_space": ColorSpaceParams,
        "flip_rotate": FlipRotateParams, "noise_injection": NoiseInjectionParams, 
        "cutmix": CutMixParams, "mixup": MixupParams,
        "dim_reducer": DimReducerParams, "hog_extractor": HOGParams, 
        "sift_extractor": SIFTParams, "orb_extractor": ORBParams,
        "cnn_embedder": CNNEmbedderParams, "vit_embedder": ViTEmbedderParams,

        # DOMAIN-SPECIFIC (8)
        "depth_cleaner": DepthCleanerParams, "depth_augmenter": DepthAugmenterParams,
        "mask_cleaner": MaskCleanerParams, "mask_augmenter": MaskAugmenterParams,
        "pointcloud_cleaner": PointCloudCleanerParams, "pointcloud_augmenter": PointCloudAugmenterParams,
        "text_tokenizer": TextTokenizerParams, "text_augmenter": TextAugmenterParams,
    }
    
    @validator('params', pre=True, always=True)
    def validate_params_against_type(cls, params: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validates the 'params' dictionary against the specific Pydantic schema for the given component 'type'.
        """
        component_type = values.get('type')
        
        if component_type in cls._PARAM_MAP and params is not None:
            try:
                # Enforce validation of the specific parameters
                return cls._PARAM_MAP[component_type](**params).dict()
            except Exception as e:
                # CRITICAL: Detailed error logging for failed validation
                raise ValueError(f"Validation failed for '{component_type}' params: {e}")
        
        return params