import logging
from pydantic import Field, validator, NonNegativeInt, conlist
from typing import List, Dict, Any, Literal, Optional

# Import Base Schemas
from .base_component_schema import BaseConfig, ComponentStepConfig

logger = logging.getLogger(__name__)

# --- 1. Cleaning Config ---

class CleaningConfig(BaseConfig):
    """
    Schema cho Image Cleaning pipeline (bước tiền xử lý bắt buộc đầu tiên).
    """
    policy_mode: Literal["default", "conditional_metadata", "adaptive_params"] = Field(
        "default", 
        description="Policy mode cho cleaning: 'default', 'conditional_metadata', or 'adaptive_params'."
    )
    
    steps: List[ComponentStepConfig] = Field(
        default_factory=list, 
        description="List of mandatory image cleaning steps (e.g., resize, normalize, color_space)."
    )

    @validator('steps')
    def validate_cleaning_steps(cls, v):
        """Rule: Ensure all cleaning steps are valid cleaner types."""
        valid_cleaners = ['resizer', 'normalizer', 'color_space']
        for step in v:
            if step.type not in valid_cleaners:
                raise ValueError(f"Cleaning pipeline only supports steps: {', '.join(valid_cleaners)}. Got: {step.type}")
        return v

# --- 2. Augmentation Config ---

class AugmentationConfig(BaseConfig):
    """
    Schema cho Augmentation pipeline (bước tùy chọn, thường chỉ cho training).
    Hỗ trợ Policy-based Augmentation như RandAugment.
    """
    policy_mode: Literal["sequential", "random_subset", "randaugment"] = Field(
        "sequential", 
        description="Mode for selecting augmenters: 'sequential', 'random_subset', or 'randaugment'."
    )
    
    n_select: Optional[NonNegativeInt] = Field(
        None, 
        description="The number of augmenters (N) to select when policy_mode is not 'sequential'."
    )
    
    magnitude: Optional[float] = Field(
        None,
        ge=0.0, 
        le=1.0, 
        description="The fixed magnitude (M) used to scale the intensity of selected augmentations in policy modes."
    )

    steps: List[ComponentStepConfig] = Field(
        [], 
        description="List of available atomic augmentation steps (Flip, Noise, CutMix, Mixup)."
    )
    
    @validator('n_select')
    def validate_n_select_policy(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Rule: Ensures n_select is a positive integer when using Policy Modes."""
        mode = values.get('policy_mode')
        
        if mode != "sequential":
            if v is not None and v == 0:
                 raise ValueError("n_select must be a positive integer or None for policy modes.")
            
        return v
    
    @validator('steps', always=True)
    def validate_augmentation_usage(cls, v: List[ComponentStepConfig], values: Dict[str, Any]) -> List[ComponentStepConfig]:
        """Rules: Checks if the policy can fulfill the selection requirement."""
        mode = values.get('policy_mode')
        n_select = values.get('n_select')
        required_steps = n_select or 1 

        if mode != "sequential" and len(v) < required_steps:
             raise ValueError(f"Policy mode '{mode}' requires at least {required_steps} step(s) to select from, but only {len(v)} steps are defined.")

        return v

# --- 3. Feature Extraction Config ---

class FeatureExtractionConfig(BaseConfig):
    """
    Schema cho Feature Extraction hoặc Embedding pipeline.
    """
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy for executing feature steps: 'default' or 'conditional_metadata'."
    )
    
    components: conlist(ComponentStepConfig, min_length=1) = Field(
        ..., 
        description="List of feature/embedding/reduction steps."
    )
    
    @validator('components')
    def validate_embedding_setup(cls, v: List[ComponentStepConfig]) -> List[ComponentStepConfig]:
        """Rule: Ensures only one deep learning Embedder (CNN/ViT) is active."""
        embedders = ['cnn_embedder', 'vit_embedder']
        embedder_count = sum(1 for step in v if step.type in embedders)
        if embedder_count > 1:
            raise ValueError("Only one deep learning embedder (cnn_embedder or vit_embedder) can be active in the feature pipeline.")
        return v