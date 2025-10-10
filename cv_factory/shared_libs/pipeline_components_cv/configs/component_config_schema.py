# cv_factory/shared_libs/ml_core/pipeline_components_cv/configs/component_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, PositiveInt, constr
from typing import List, Dict, Any, Union, Literal, Optional, Tuple
import logging
import cv2 # Used for interpolation constants

logger = logging.getLogger(__name__)

# Assuming BaseConfig is available
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


# --- 1. Schemas for Stateless Cleaning Adapters ---

class ResizerParams(BaseModel):
    """Parameters for CVResizer."""
    width: PositiveInt = Field(..., description="Target width.")
    height: PositiveInt = Field(..., description="Target height.")
    interpolation: Literal[cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST] = Field(
        cv2.INTER_AREA, description="OpenCV interpolation method."
    )

class NormalizerParams(BaseModel):
    """Parameters for CVNormalizer."""
    mean: Union[float, List[float]] = Field(..., description="Normalization mean (float or list of 3 floats).")
    std: Union[float, List[float]] = Field(..., description="Normalization standard deviation.")
    
class ColorSpaceParams(BaseModel):
    """Parameters for CVColorSpaceCleaner."""
    conversion_code: Literal["BGR2RGB", "RGB2BGR", "BGR2GRAY", "RGB2GRAY"] = Field(..., description="Color conversion code.")


# --- 2. Schemas for Feature Extraction/Embedding Adapters ---

class DimReducerParams(BaseModel):
    """Parameters for CVDimReducer (Stateful)."""
    method: Literal["pca", "umap"] = Field(..., description="Reduction method.")
    n_components: PositiveInt = Field(2, description="Number of output components.")

class CNNEmbedderParams(BaseModel):
    """Parameters for CVCNNEmbedder (Stateful/Model)."""
    model_name: str = Field(..., description="Torchvision model name (e.g., 'resnet18').")
    pretrained: bool = Field(True, description="Load pretrained weights.")
    remove_head: bool = Field(True, description="Remove the classification head for feature extraction.")

class ViTEmbedderParams(BaseModel):
    """Parameters for CVViTEmbedder (Stateful/Model)."""
    model_name: str = Field(..., description="Hugging Face ViT model ID.")
    pretrained: bool = Field(True, description="Load pretrained weights.")

# --- 3. Schemas for Augmentation Adapters ---

class FlipRotateParams(BaseModel):
    """Parameters for CVFlipRotate."""
    flip_prob: Field(default=0.5, ge=0.0, le=1.0) = Field(description="Probability of applying flip.")
    rotate_limit: NonNegativeInt = Field(15, description="Max rotation angle in degrees.")

class NoiseInjectionParams(BaseModel):
    """Parameters for CVNoiseInjection."""
    noise_type: Literal["gaussian", "salt_pepper"] = Field("gaussian", description="Type of noise to inject.")
    strength: Field(default=0.05, ge=0.0) = Field(description="Intensity/Strength of the noise.")

class CutMixParams(BaseModel):
    """Parameters for CVCutMix."""
    alpha: Field(default=1.0, ge=0.0) = Field(description="Beta distribution alpha parameter.")

class MixupParams(BaseModel):
    """Parameters for CVMixup."""
    alpha: Field(default=0.2, ge=0.0) = Field(description="Beta distribution alpha parameter.")


# --- 4. Schemas for Feature Extractor Adapters (Classical) ---

class HOGParams(BaseModel):
    """Parameters for CVHOGExtractor."""
    orientations: PositiveInt = Field(9, description="Number of orientation bins.")
    pixels_per_cell: Tuple[PositiveInt, PositiveInt] = Field((8, 8), description="Size of the cell in pixels.")

class SIFTParams(BaseModel):
    """Parameters for CVSIFTExtractor."""
    nfeatures: NonNegativeInt = Field(0, description="Max number of features to retain.")

class ORBParams(BaseModel):
    """Parameters for CVORBExtractor."""
    nfeatures: PositiveInt = Field(500, description="Max number of features to retain.")
    scaleFactor: Field(default=1.2, ge=1.0) = Field(description="Pyramid scale factor.")


# --- 5. Master Step Configuration (The Final Product) ---

# We define a single component configuration that uses a dynamic validation method (Sub-schema delegation).
class PipelineStepConfig(BaseConfig):
    """
    Master schema for a single step in the preprocessing pipeline. 
    It delegates validation based on the 'type' field.
    """
    type: constr(to_lower=True) = Field(..., description="The Adapter component type (e.g., 'resizer', 'cnn_embedder').")
    
    @validator('params', pre=True)
    def validate_params_against_type(cls, params, values):
        """
        Rule: Validate 'params' dictionary against the specific schema for the given 'type'.
        """
        component_type = values.get('type')
        
        # Mapping component type to its specific Pydantic parameters schema
        PARAM_MAP = {
            # CLEANING
            "resizer": ResizerParams,
            "normalizer": NormalizerParams,
            "color_space": ColorSpaceParams,
            
            # AUGMENTATION <<< NEW ENTRIES >>>
            "flip_rotate": FlipRotateParams,
            "noise_injection": NoiseInjectionParams,
            "cutmix": CutMixParams,
            "mixup": MixupParams,
            
            # FEATURE EXTRACTION & DIM REDUCTION
            "dim_reducer": DimReducerParams,
            "hog_extractor": HOGParams,
            "sift_extractor": SIFTParams,
            "orb_extractor": ORBParams,

            # DEEP LEARNING EMBEDDERS
            "cnn_embedder": CNNEmbedderParams,
            "vit_embedder": ViTEmbedderParams,
        }
        
        if component_type in PARAM_MAP:
            try:
                # Enforce validation of the specific parameters
                return PARAM_MAP[component_type](**params).dict()
            except Exception as e:
                # CRITICAL: Detailed error logging for failed validation
                raise ValueError(f"Validation failed for '{component_type}' params: {e}")
        
        # If no specific schema is found (e.g., for custom components), return original params
        return params