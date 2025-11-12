# shared_libs/ml_core/pipeline_components_cv/configs/schemas/depth_schemas.py

from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt

# --- Depth Domain Schemas ---
class DepthCleanerParams(BaseModel):
    rgb_path_key: str = Field('rgb_path')
    depth_path_key: str = Field('depth_path')
    min_depth: float = Field(0.1, ge=0.0)
    max_depth: float = Field(10.0, gt=0.0)
    normalize_depth: bool = Field(True)

class DepthAugmenterParams(BaseModel):
    p_augment: float = Field(0.3, ge=0.0, le=1.0)
    jitter_variance: float = Field(0.01, ge=0.0)
    max_rotation_deg: float = Field(5.0, ge=0.0)