# shared_libs/ml_core/pipeline_components_cv/configs/schemas/mask_schemas.py

from pydantic import BaseModel, Field, NonNegativeInt

# --- Mask Domain Schemas ---
class MaskCleanerParams(BaseModel):
    rgb_path_key: str = Field('rgb_path')
    mask_path_key: str = Field('mask_path')
    min_mask_area: NonNegativeInt = Field(100)
    apply_erosion: bool = Field(False)

class MaskAugmenterParams(BaseModel):
    p_augment: float = Field(0.4, ge=0.0, le=1.0)
    preserve_mask_shape: bool = Field(True)