import logging
from pydantic import Field, validator, conlist
from typing import List, Dict, Any, Literal, Optional

# Import Base Schemas
from .base_component_schema import BaseConfig, ComponentStepConfig

logger = logging.getLogger(__name__)

class VideoProcessingConfig(BaseConfig):
    """
    Schema cho toàn bộ Video Processing pipeline, quản lý Video Cleaning và Frame Sampling.
    """
    
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy for video execution: 'default' or 'conditional_metadata'."
    )
    
    # 1. Video Cleaning Steps (4D -> 4D)
    cleaners: List[ComponentStepConfig] = Field(
        default_factory=list,
        description="List of video cleaning steps (e.g., video_resizer, frame_rate_adjuster)."
    )

    # 2. Frame Sampling Steps (CRITICAL: 4D -> List[3D])
    samplers: conlist(ComponentStepConfig, min_length=1) = Field(
        ...,
        description="List of Frame Samplers (e.g., uniform_sampler, keyframe_extractor, policy_sampler). Must have at least one step."
    )
    
    @validator('samplers')
    def validate_sampler_type(cls, v: List[ComponentStepConfig]) -> List[ComponentStepConfig]:
        """Rule: Ensure all sampler steps are valid sampler types."""
        valid_samplers = ['uniform_sampler', 'keyframe_extractor', 'motion_aware_sampler', 'policy_sampler'] 
        for step in v:
            if step.type not in valid_samplers:
                raise ValueError(f"Video Sampler pipeline only supports steps: {', '.join(valid_samplers)}. Got: {step.type}")
        return v
    
    @validator('cleaners')
    def validate_cleaner_type(cls, v: List[ComponentStepConfig]) -> List[ComponentStepConfig]:
        """Rule: Ensure all cleaner steps are valid video cleaner types."""
        valid_cleaners = ['video_resizer', 'frame_rate_adjuster', 'video_noise_reducer', 'motion_stabilizer'] 
        for step in v:
            if step.type not in valid_cleaners:
                raise ValueError(f"Video Cleaner pipeline only supports steps: {', '.join(valid_cleaners)}. Got: {step.type}")
        return v

    @validator('samplers', always=True)
    def check_sampling_count(cls, v: List[ComponentStepConfig], values: Dict[str, Any]) -> List[ComponentStepConfig]:
        """Rule: If PolicySampler is used, ensure it is the only component."""
        
        policy_sampler_count = sum(1 for step in v if step.type == 'policy_sampler')
        
        if policy_sampler_count > 1:
            raise ValueError("Only one 'policy_sampler' instance is allowed in the samplers list.")
        
        if policy_sampler_count == 1 and len(v) > 1:
             raise ValueError("If 'policy_sampler' is used, it must be the only component in the samplers list.")
             
        return v