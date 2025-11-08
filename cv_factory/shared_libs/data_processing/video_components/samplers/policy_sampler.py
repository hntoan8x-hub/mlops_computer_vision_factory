# shared_libs/data_processing/video_components/samplers/policy_sampler.py

import logging
from typing import Dict, Any, Union, List, Optional, Type, Literal

# Import Abstractions and Data Types
from shared_libs.data_processing.video_components._base.base_frame_sampler import BaseFrameSampler, VideoData, ImageData

# Import Concrete Samplers (Adaptees)
from shared_libs.data_processing.video_components.samplers.uniform_frame_sampler import UniformFrameSampler
from shared_libs.data_processing.video_components.samplers.key_frame_extractor import KeyFrameExtractor
from shared_libs.data_processing.video_components.samplers.motion_aware_sampler import MotionAwareSampler 

logger = logging.getLogger(__name__)

# Supported policies for the Policy Sampler
SamplerPolicy = Literal["uniform", "keyframe", "motion_aware"]

class PolicySampler(BaseFrameSampler):
    """
    Adapter component that selects and delegates sampling to a specific atomic sampler 
    (Uniform, KeyFrame, MotionAware) based on a configured policy type.

    This acts as a Façade/Router for frame sampling logic.
    """
    
    _SAMPLER_MAP: Dict[str, Type[BaseFrameSampler]] = {
        "uniform": UniformFrameSampler,
        "keyframe": KeyFrameExtractor,
        "motion_aware": MotionAwareSampler,
    }

    def __init__(self, policy_type: SamplerPolicy = "uniform", config_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the Policy Sampler and the underlying selected Sampler.

        Args:
            policy_type (SamplerPolicy): The strategy to use for frame sampling.
            config_params (Optional[Dict[str, Any]]): Configuration parameters specific 
                                                      to the chosen policy (e.g., 'threshold' for keyframe).

        Raises:
            ValueError: If the policy_type is unsupported.
            RuntimeError: If instantiation of the underlying sampler fails.
        """
        if policy_type not in self._SAMPLER_MAP:
            supported = ", ".join(self._SAMPLER_MAP.keys())
            raise ValueError(f"Unsupported sampling policy: {policy_type}. Supported policies: {supported}.")
            
        self.policy_type = policy_type
        self.config_params = config_params or {}
        
        # Instantiate the selected concrete sampler (Adapter Pattern)
        SelectedSampler = self._SAMPLER_MAP[policy_type]
        try:
            self._active_sampler = SelectedSampler(**self.config_params)
        except Exception as e:
            logger.error(f"Failed to initialize underlying sampler '{policy_type}' with params {self.config_params}. Error: {e}")
            raise RuntimeError(f"Policy Sampler instantiation failed: {e}")
        
        logger.info(f"Initialized Policy Sampler using '{policy_type}' strategy.")

    def sample(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Delegates the frame sampling execution to the currently active atomic sampler.

        Args:
            video (VideoData): The input video sequence(s).
            metadata (Optional[Dict[str, Any]]): Metadata (passed through).
            **kwargs: Additional keyword arguments (passed through).

        Returns:
            ImageData: The list of 3D image frames extracted by the chosen policy.
        """
        # Delegation: The Policy Sampler simply calls the sample method of the chosen component
        return self._active_sampler.sample(video, metadata=metadata, **kwargs)

    # NOTE: In a full MLOps cycle, save/load would be implemented here to save 
    # the state (policy type and params) of the Policy Sampler, and delegate 
    # the active sampler's state if it were stateful (e.g., adaptive sampler).