# shared_libs/ml_core/pipeline_components_cv/configs/schemas/pointcloud_schemas.py

from pydantic import BaseModel, Field, PositiveInt
from typing import Optional

# --- Point Cloud Domain Schemas ---
class PointCloudCleanerParams(BaseModel):
    path_key: str = Field('pcd_path')
    voxel_size: float = Field(0.05, gt=0.0)
    normalize_coords: bool = Field(True)
    max_points: Optional[PositiveInt] = Field(None)

class PointCloudAugmenterParams(BaseModel):
    p_augment: float = Field(0.5, ge=0.0, le=1.0)
    max_translation: float = Field(0.1, ge=0.0)
    apply_jitter: bool = Field(True)