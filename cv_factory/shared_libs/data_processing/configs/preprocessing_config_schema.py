# cv_factory/shared_libs/data_processing/configs/preprocessing_config_schema.py

from pydantic import Field, validator, NonNegativeInt, conlist, BaseModel, constr
from typing import List, Dict, Any, Union, Literal, Optional

# Assuming BaseConfig is available from shared_libs/ml_core/configs/base_config_schema.py
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")

# --- 1. Atomic Component Step Schema ---

class ComponentStepConfig(BaseConfig):
    """Schema for a single step within a pipeline (must correspond to a ComponentFactory key)."""
    type: constr(to_lower=True) = Field(..., description="The type/name of the component (e.g., 'resizer', 'normalizer').")
    
    @validator('type')
    def validate_known_component(cls, v):
        # Enforce that the component type is one that the ComponentFactory can instantiate
        supported = [
            'resizer', 'normalizer', 'color_space', 
            'flip_rotate', 'noise_injection', 'cutmix', 'mixup',
            'dim_reducer', 
            'cnn_embedder', 'vit_embedder',
            'hog_extractor', 'sift_extractor', 'orb_extractor'
        ]
        if v not in supported:
            raise ValueError(f"Unsupported component type: {v}. Must be one of {supported}.")
        return v


# --- 2. Module Schemas (Groups of Steps) ---

class CleaningConfig(BaseConfig):
    """Schema for the Cleaning module (mandatory preprocessing steps)."""
    steps: conlist(ComponentStepConfig, min_length=1) = Field(..., description="Ordered list of cleaning steps (must have at least one step).")
    
    @validator('steps')
    def must_include_resizer(cls, v):
        """Rule: Enforce that resizing is typically included in cleaning for most CV models."""
        step_types = [step.type for step in v]
        if 'resizer' not in step_types:
            logger.warning("Warning: Resizer component missing. Ensure inputs have uniform size.")
        return v
    
class AugmentationConfig(BaseConfig):
    """Schema for the Augmentation module (only used during training)."""
    steps: conlist(ComponentStepConfig, min_length=0) = Field([], description="Ordered list of augmentation steps (optional).")
    
    @validator('steps')
    def check_augmentation_types(cls, v):
        """Rule: Augmentation steps must only include known augmentation types."""
        allowed_aug = ['flip_rotate', 'noise_injection', 'cutmix', 'mixup']
        for step in v:
            if step.type not in allowed_aug:
                raise ValueError(f"Invalid augmentation step '{step.type}'. Only {allowed_aug} are allowed here.")
        return v

class FeatureExtractionConfig(BaseConfig):
    """Schema for Feature Extraction or Embedding."""
    # This schema groups all types of feature generation and optimization
    
    # Allows a single component or a list of components
    components: conlist(ComponentStepConfig, min_length=1) = Field(..., description="List of feature/embedding/reduction steps.")
    
    @validator('components')
    def validate_embedding_setup(cls, v):
        """Rule: Ensure only one Embedder (CNN/ViT) is active in the list."""
        embedders = ['cnn_embedder', 'vit_embedder']
        embedder_count = sum(1 for step in v if step.type in embedders)
        if embedder_count > 1:
            raise ValueError("Only one deep learning embedder (cnn_embedder or vit_embedder) can be active in the feature pipeline.")
        return v


# --- 3. Master Processing Schema ---

class ProcessingConfig(BaseConfig):
    """Master schema for the entire Data Processing and Feature Engineering pipeline."""
    
    cleaning: CleaningConfig = Field(..., description="The mandatory cleaning pipeline.")
    augmentation: AugmentationConfig = Field(..., description="The optional augmentation pipeline.")
    feature_engineering: FeatureExtractionConfig = Field(..., description="Configuration for feature generation and optimization.")

    @validator('augmentation')
    def validate_augmentation_usage(cls, v, values):
        """Rule: Augmentation must be disabled for inference context."""
        # Assuming the top-level OrchestratorConfig specifies the context.
        # This check should ideally be done in the Orchestrator Config (Step 3).
        if not v.enabled and len(v.steps) > 0:
            raise ValueError("Augmentation steps must be empty if 'enabled' is False.")
        return v