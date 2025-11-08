# shared_libs/data_processing/video_components/samplers/frame_sampler_factory.py

import logging
from typing import Dict, Any, Type, Optional, Union

# Import Abstraction
from shared_libs.data_processing.video_components._base.base_frame_sampler import BaseFrameSampler

# Import Atomic Components (Newly Implemented)
from shared_libs.data_processing.video_components.samplers.uniform_frame_sampler import UniformFrameSampler
from shared_libs.data_processing.video_components.samplers.key_frame_extractor import KeyFrameExtractor 
from shared_libs.data_processing.video_components.samplers.motion_aware_sampler import MotionAwareSampler 
from shared_libs.data_processing.video_components.samplers.policy_sampler import PolicySampler # The Facade Sampler

logger = logging.getLogger(__name__)

class FrameSamplerFactory:
    """
    A factory class for creating video frame sampling components.

    This factory centralizes the creation logic for Bridge Components, including 
    the PolicySampler façade for adaptive sampling.
    """
    _SAMPLER_MAP: Dict[str, Type[BaseFrameSampler]] = {
        "uniform_sampler": UniformFrameSampler,
        "keyframe_extractor": KeyFrameExtractor,
        "motion_aware_sampler": MotionAwareSampler,
        "policy_sampler": PolicySampler, # The Policy Facade itself
    }

    @classmethod
    def create(cls, sampler_type: str, config: Optional[Dict[str, Any]] = None) -> BaseFrameSampler:
        """
        Creates and returns a frame sampler instance based on its type.

        Args:
            sampler_type (str): The type of sampler to create (e.g., "uniform_sampler", "policy_sampler").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseFrameSampler: An instance of the requested sampler.

        Raises:
            ValueError: If the specified sampler_type is not supported.
            RuntimeError: If instantiation fails due to bad parameters.
        """
        config = config or {}
        sampler_type_lower = sampler_type.lower()
        sampler_cls = cls._SAMPLER_MAP.get(sampler_type_lower)
        
        if not sampler_cls:
            supported_samplers = ", ".join(cls._SAMPLER_MAP.keys())
            logger.error(f"Unsupported frame sampler type: '{sampler_type}'. Supported types are: {supported_samplers}")
            raise ValueError(f"Unsupported frame sampler type: '{sampler_type}'. Supported types are: {supported_samplers}")

        logger.info(f"Creating instance of {sampler_cls.__name__}...")
        try:
            # Hardening: Wrap instantiation for robust error handling
            # Note: For PolicySampler, config will contain 'policy_type' and specific params.
            return sampler_cls(**config)
        except Exception as e:
            logger.error(f"Factory failed to instantiate {sampler_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{sampler_type}': {e}")