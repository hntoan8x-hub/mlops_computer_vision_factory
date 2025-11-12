# cv_factory/shared_libs/data_processing/configs/preprocessing_config_schema.py

import logging
from pydantic import Field, validator, NonNegativeInt, conlist, BaseModel, constr
from typing import List, Dict, Any, Union, Literal, Optional

logger = logging.getLogger(__name__)

# NOTE: Giả định BaseConfig đã được Hardening với extra="forbid" trong môi trường Production
class BaseConfig(BaseModel):
    class Config:
        # Sử dụng 'forbid' trong Production để tăng tính an toàn cấu hình
        extra = "forbid" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")

# --- 1. Atomic Component Step Schema ---

class ComponentStepConfig(BaseConfig):
    """Schema for a single step within a pipeline (must correspond to a ComponentFactory key)."""
    type: constr(to_lower=True) = Field(..., description="The type/name of the component (e.g., 'resizer', 'normalizer').")
    
    @validator('type')
    def validate_known_component(cls, v):
        """Validates that the component type is supported."""
        supported = [
            'resizer', 'normalizer', 'color_space', 
            'flip_rotate', 'noise_injection', 'cutmix', 'mixup',
            'dim_reducer', 
            'cnn_embedder', 'vit_embedder',
            'hog_extractor', 'sift_extractor', 'orb_extractor'
            # NEW DEPTH COMPONENTS
            , 'loader', 'normalizer', 'augmenter', 'validator', 'converter' # Tên ngắn cho Depth
        ]
        if v not in supported:
            raise ValueError(f"Unknown component type: {v}. Must be one of: {', '.join(supported)}")
        return v

    @validator('params')
    def validate_normalizer_params(cls, v: Optional[Dict[str, Any]], values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enforces consistency rules specifically for the 'normalizer' component."""
        type_ = values.get('type')
        if type_ == 'normalizer':
            if not v or 'mean' not in v or 'std' not in v:
                raise ValueError("Normalizer component requires 'mean' and 'std' parameters to be defined.")
            
            mean = v.get('mean')
            std = v.get('std')
            
            if type(mean) is not type(std):
                 raise ValueError("Normalizer: 'mean' and 'std' must be of the same type (e.g., both list or both float).")

            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
                if len(mean) != len(std):
                    raise ValueError(f"Normalizer: 'mean' ({len(mean)} elements) and 'std' ({len(std)} elements) must have the same length.")
        
        # Hardening cho Depth Normalizer Config
        if type_ == 'normalizer' and 'method' in v and v['method'] == 'minmax':
             if 'scale' not in v or not isinstance(v['scale'], list) or len(v['scale']) != 2:
                 raise ValueError("Depth Normalizer ('minmax' method) requires 'scale' parameter as a list of two floats [min, max].")

        return v

# --- 2. Module Schemas (Groups of Steps) ---

class CleaningConfig(BaseConfig):
    """
    Schema for the image cleaning pipeline (mandatory first step).
    Includes policy mode for adaptive/conditional execution based on metadata.
    """
    # HARDENING: Add policy_mode for Adaptive Cleaning
    policy_mode: Literal["default", "conditional_metadata", "adaptive_params"] = Field(
        "default", 
        description="Policy for executing cleaning steps: 'default', 'conditional_metadata', or 'adaptive_params'."
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

    @validator('steps', always=True)
    def must_include_resizer(cls, v):
        """Rule: Enforce that resizing is typically included for uniform inputs."""
        step_types = [step.type for step in v]
        if 'resizer' not in step_types:
            logging.warning("Warning: Resizer component missing. Ensure inputs have uniform size.")
        return v
    
class AugmentationConfig(BaseConfig):
    """
    Schema for the Augmentation pipeline (optional steps), specifically supporting 
    Policy-based Augmentation like RandAugment.
    """
    # 1. Policy Mode: Controls how steps are selected and executed.
    policy_mode: Literal["sequential", "random_subset", "randaugment"] = Field(
        "sequential", 
        description="Mode for selecting augmenters: 'sequential', 'random_subset', or 'randaugment'."
    )
    
    # 2. N (Number of Steps): Required for policy modes.
    n_select: Optional[NonNegativeInt] = Field(
        None, 
        description="The number of augmenters (N) to select when policy_mode is not 'sequential'."
    )
    
    # 3. M (Magnitude): CRITICAL for RandAugment. Defines the fixed intensity for selected operations.
    magnitude: Optional[float] = Field(
        None,
        ge=0.0,  # Greater than or equal to 0.0
        le=1.0,  # Less than or equal to 1.0 (typical range for scaling)
        description="The fixed magnitude (M) used to scale the intensity of selected augmentations in policy modes."
    )

    # 4. Atomic Augmentation Steps.
    steps: List[ComponentStepConfig] = Field(
        [], 
        description="List of available atomic augmentation steps (Flip, Noise, CutMix, Mixup)."
    )
    
    @validator('n_select')
    def validate_n_select_policy(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """
        Rule: Ensures n_select is a positive integer when using Policy Modes.
        """
        mode = values.get('policy_mode')
        
        if mode != "sequential":
            # Hardening: If policy mode is active, n_select must be set and > 0, 
            # OR we allow None and let the Controller default it (e.g., to 2).
            if v is not None and v == 0:
                 raise ValueError("n_select must be a positive integer or None for policy modes.")
            
        return v
    
    @validator('magnitude')
    def validate_magnitude_policy(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        """
        Rule: Ensures magnitude is strictly defined when using RandAugment policy.
        """
        mode = values.get('policy_mode')
        
        if mode == "randaugment" and v is None:
            # Hardening: RandAugment's core feature is the fixed magnitude. 
            # It must be defined for this mode (or defaulted if not None).
            # We allow None if the Orchestrator/Controller is expected to provide a default.
            logging.warning("Magnitude is recommended to be explicitly set when policy_mode is 'randaugment'.")

        return v

    @validator('steps', always=True)
    def validate_augmentation_usage(cls, v: List[ComponentStepConfig], values: Dict[str, Any]) -> List[ComponentStepConfig]:
        """
        Rules: Checks if augmentation is enabled and if the policy can fulfill the selection requirement.
        """
        # Rule 1: Steps must be empty if the entire pipeline is disabled.
        if 'enabled' in values and not values['enabled'] and len(v) > 0:
            raise ValueError("Augmentation steps must be empty if 'enabled' is False.")
        
        # Rule 2: If policy mode is used, there must be enough steps to select from.
        mode = values.get('policy_mode')
        n_select = values.get('n_select')
        required_steps = n_select or 1 # Assume default selection is at least 1 if not specified

        if mode != "sequential" and len(v) < required_steps:
             raise ValueError(f"Policy mode '{mode}' requires at least {required_steps} step(s) to select from, but only {len(v)} steps are defined.")

        return v
# 
class FeatureExtractionConfig(BaseConfig):
    """
    Schema for the Feature Extraction or Embedding pipeline.
    
    Includes policy mode for conditional/adaptive execution and governance rules 
    for deep learning embedders.
    """
    # HARDENING: Policy Mode for Conditional/Adaptive Execution
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy for executing feature steps: 'default' (run all enabled) or 'conditional_metadata' (skip/select based on metadata)."
    )
    
    components: conlist(ComponentStepConfig, min_length=1) = Field(
        ..., 
        description="List of feature/embedding/reduction steps."
    )
    
    @validator('components')
    def validate_embedding_setup(cls, v: List[ComponentStepConfig]) -> List[ComponentStepConfig]:
        """
        Rule: Ensures only one deep learning Embedder (CNN/ViT) is active in the list.
        
        Args:
            v (List[ComponentStepConfig]): The list of feature components.

        Returns:
            List[ComponentStepConfig]: The validated list of feature components.
            
        Raises:
            ValueError: If more than one deep learning embedder is active.
        """
        embedders = ['cnn_embedder', 'vit_embedder']
        embedder_count = sum(1 for step in v if step.type in embedders)
        if embedder_count > 1:
            raise ValueError("Only one deep learning embedder (cnn_embedder or vit_embedder) can be active in the feature pipeline.")
        return v

    @validator('components', always=True)
    def validate_components_usage(cls, v: List[ComponentStepConfig], values: Dict[str, Any]) -> List[ComponentStepConfig]:
        """
        Rules: Checks if feature extraction is enabled and components are defined.
        """
        # Rule: Components list must be empty if the entire feature pipeline is disabled.
        if 'enabled' in values and not values['enabled'] and len(v) > 0:
            raise ValueError("Component steps must be empty if 'enabled' is False.")

        return v

# --- NEW INTEGRATION: Depth Processing Schema ---

class DepthProcessingConfig(BaseConfig):
    """
    Schema cho toàn bộ Depth Processing pipeline, quản lý Load, Normalization, 
    Augmentation và Validation cho cặp RGB/Depth.
    """
    # Policy mode cho Depth: Có thể mở rộng sau này
    policy_mode: Literal["default"] = Field(
        "default",
        description="Policy for depth processing: 'default' sequential execution."
    )
    
    # Các bước được định nghĩa bằng key thay vì list 'steps' để dễ dàng truy cập trong Orchestrator
    loader: ComponentStepConfig = Field(
        ...,
        description="Cấu hình cho DepthLoader (tải và căn chỉnh RGB/Depth). Loại: 'loader'."
    )
    
    normalizer: ComponentStepConfig = Field(
        ...,
        description="Cấu hình cho DepthNormalizer (chuẩn hóa giá trị độ sâu). Loại: 'normalizer'."
    )
    
    augmenter: Optional[ComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho DepthAugmenter (lật, thêm nhiễu). Loại: 'augmenter'."
    )
    
    validator: Optional[ComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho DepthValidator (kiểm tra phạm vi, chất lượng). Loại: 'validator'."
    )
    
    converter: Optional[ComponentStepConfig] = Field(
        None,
        description="Cấu hình tùy chọn cho DisparityConverter (Depth <-> Disparity). Loại: 'converter'."
    )
    
    @validator('loader', 'normalizer', 'augmenter', 'validator', 'converter', always=True)
    def validate_depth_component_types(cls, v: Optional[ComponentStepConfig], values: Dict[str, Any], field: Any) -> Optional[ComponentStepConfig]:
        """Rule: Đảm bảo các component Depth có type đúng."""
        if v is None or v.type is None:
            return v
        
        expected_type = field.name
        # Đối với loader, normalizer, etc., type phải khớp với tên trường
        if v.type.lower() != expected_type:
            raise ValueError(
                f"DepthProcessingConfig field '{field.name}' requires type '{expected_type}', but got '{v.type}'."
            )
        return v


# --- Video Processing schema
class VideoProcessingConfig(BaseConfig):
    """
    Schema for the entire Video Processing pipeline, managing cleaning steps and 
    the critical Frame Sampling bridge component.
    """
    
    # HARDENING: Policy Mode for Video (Simple conditional execution)
    policy_mode: Literal["default", "conditional_metadata"] = Field(
        "default",
        description="Policy for video execution: 'default' sequential execution or 'conditional_metadata' adaptive execution."
    )
    
    # 1. Video Cleaning Steps (4D -> 4D)
    cleaners: List[ComponentStepConfig] = Field(
        default_factory=list,
        description="List of video cleaning steps (e.g., video_resizer, frame_rate_adjuster, motion_stabilizer)."
    )

    # 2. Frame Sampling Steps (CRITICAL: 4D -> List[3D])
    # Sử dụng conlist với min_length=1 để đảm bảo luôn có Sampler nếu luồng Video được bật
    samplers: conlist(ComponentStepConfig, min_length=1) = Field(
        ...,
        description="List of Frame Samplers (e.g., uniform_sampler, keyframe_extractor, policy_sampler). Must have at least one step."
    )
    
    @validator('samplers')
    def validate_sampler_type(cls, v: List[ComponentStepConfig]) -> List[ComponentStepConfig]:
        """Rule: Ensure all sampler steps are valid sampler types managed by FrameSamplerFactory."""
        valid_samplers = ['uniform_sampler', 'keyframe_extractor', 'motion_aware_sampler', 'policy_sampler'] 
        for step in v:
            if step.type not in valid_samplers:
                raise ValueError(f"Video Sampler pipeline only supports steps: {', '.join(valid_samplers)}. Got: {step.type}")
        return v
    
    @validator('cleaners')
    def validate_cleaner_type(cls, v: List[ComponentStepConfig]) -> List[ComponentStepConfig]:
        """Rule: Ensure all cleaner steps are valid video cleaner types managed by VideoCleanerFactory."""
        valid_cleaners = ['video_resizer', 'frame_rate_adjuster', 'video_noise_reducer', 'motion_stabilizer'] 
        for step in v:
            if step.type not in valid_cleaners:
                raise ValueError(f"Video Cleaner pipeline only supports steps: {', '.join(valid_cleaners)}. Got: {step.type}")
        return v

    @validator('samplers', always=True)
    def check_sampling_count(cls, v: List[ComponentStepConfig], values: Dict[str, Any]) -> List[ComponentStepConfig]:
        """Rule: If a sampling policy (like PolicySampler) is used, ensure only one main sampler step is defined."""
        
        policy_sampler_count = sum(1 for step in v if step.type == 'policy_sampler')
        
        # Hardening: If PolicySampler is used, it should be the only step in the samplers list 
        # (as it acts as a Façade/Router for the actual sampling logic).
        if policy_sampler_count > 1:
            raise ValueError("Only one 'policy_sampler' instance is allowed in the samplers list.")
        
        if policy_sampler_count == 1 and len(v) > 1:
             raise ValueError("If 'policy_sampler' is used, it must be the only component in the samplers list.")
             
        return v
    
    
class MaskProcessingConfig(BaseModel):
    """Schema for the Mask/Label Map Processing pipeline."""
    enabled: bool = Field(False, description="Enables/disables the entire mask processing pipeline.")
    policy_mode: constr(to_lower=True) = Field("default", description="Execution policy ('default' or 'conditional_metadata').")
    steps: List[ComponentStepConfig] = Field(..., description="Ordered list of mask processing steps.")

# --- 4. Master Processing Schema ---

class ProcessingConfig(BaseConfig):
    """Master schema for the entire Data Processing and Feature Engineering pipeline."""
    
    cleaning: CleaningConfig = Field(..., description="The mandatory image cleaning pipeline.")
    augmentation: AugmentationConfig = Field(..., description="The optional augmentation pipeline.")
    feature_engineering: FeatureExtractionConfig = Field(..., description="Configuration for feature generation and optimization.")
    
    # NEW INTEGRATION: Video Processing Layer (Optional)
    video_processing: Optional[VideoProcessingConfig] = Field(
        None,
        description="Configuration for the optional video processing and frame sampling pipeline."
    )
    
    # NEW INTEGRATION: Depth Processing Layer (Optional)
    depth_processing: Optional[DepthProcessingConfig] = Field(
        None,
        description="Configuration for the optional depth map processing pipeline (RGB/Depth pairs)."
    )
    
     # NEW INTEGRATION: Mask Processing Layer (Optional)
    mask_processing: Optional[MaskProcessingConfig] = Field(
        None,
        description="Configuration for the optional mask/label map processing pipeline (e.g., loading, normalization)."
    )
    
    feature_engineering: FeatureExtractionConfig = Field(..., description="Configuration for feature generation and optimization.")

    @validator('augmentation')
    def validate_augmentation_usage(cls, v: AugmentationConfig, values: Dict[str, Any]) -> AugmentationConfig:
        """
        Rule: Enforces that augmentation steps must be empty if 'enabled' is False, 
        typically for the Inference context.
        
        Args:
            v (AugmentationConfig): The augmentation configuration.
            values (Dict[str, Any]): All previously validated field values.

        Returns:
            AugmentationConfig: The validated augmentation configuration.
            
        Raises:
            ValueError: If augmentation is disabled but steps are present.
        """
        if not v.enabled and len(v.steps) > 0:
            raise ValueError("Augmentation steps must be empty if 'enabled' is False (typically for Inference/Serving).")
        return v