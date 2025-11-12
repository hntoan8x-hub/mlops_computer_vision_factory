# shared_libs/ml_core/pipeline_components_cv/configs/schemas/core_cv_schemas.py

from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt
from typing import List, Union, Literal, Optional, Tuple
import cv2 

# --- 1. Cleaning Adapters ---
class ResizerParams(BaseModel):
    width: PositiveInt = Field(..., description="Target width.")
    height: PositiveInt = Field(..., description="Target height.")
    interpolation: Literal[cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST] = Field(
        cv2.INTER_AREA, description="OpenCV interpolation method."
    )

class NormalizerParams(BaseModel):
    mean: List[float] = Field(..., description="Per-channel mean.")
    std: List[float] = Field(..., description="Per-channel standard deviation.")

class ColorSpaceParams(BaseModel):
    conversion_code: Literal["BGR2RGB", "RGB2BGR", "BGR2GRAY", "RGB2GRAY"] = Field(..., description="Color conversion code.")

# --- 2. Augmentation Adapters ---
class FlipRotateParams(BaseModel):
    p_flip_h: float = Field(0.5, ge=0.0, le=1.0)
    p_rotate_90: float = Field(0.0, ge=0.0, le=1.0)

class NoiseInjectionParams(BaseModel):
    noise_type: Literal['gaussian', 'salt_pepper'] = Field('gaussian')
    intensity: float = Field(0.1, ge=0.0)

class CutMixParams(BaseModel):
    alpha: float = Field(1.0, gt=0.0)
    p: float = Field(0.5, ge=0.0, le=1.0)

class MixupParams(BaseModel):
    alpha: float = Field(1.0, gt=0.0)
    p: float = Field(0.5, ge=0.0, le=1.0)

# --- 3. Feature Extraction & Dim Reduction Adapters ---
class DimReducerParams(BaseModel):
    method: Literal['pca', 'tsne'] = Field('pca')
    n_components: PositiveInt = Field(64)

class HOGParams(BaseModel):
    win_size: Tuple[PositiveInt, PositiveInt] = Field((64, 128))
    block_size: Tuple[PositiveInt, PositiveInt] = Field((16, 16))

class SIFTParams(BaseModel):
    n_features: NonNegativeInt = Field(0)

class ORBParams(BaseModel):
    n_features: PositiveInt = Field(500)
    scale_factor: float = Field(1.2, gt=1.0)

# --- 4. Deep Learning Embedders ---
class CNNEmbedderParams(BaseModel):
    model_name: str
    output_dim: PositiveInt = Field(128)
    freeze_weights: bool = Field(True)

class ViTEmbedderParams(BaseModel):
    model_name: str
    output_dim: PositiveInt = Field(768)