# shared_libs/data_processing/video_components/cleaners/video_cleaner_factory.py

import logging
from typing import Dict, Any, Type, Optional, Union

# Import Abstraction
from shared_libs.data_processing.video_components._base.base_video_cleaner import BaseVideoCleaner

# Import Atomic Components (Newly Implemented)
from shared_libs.data_processing.video_components.cleaners.video_frame_resizer import VideoFrameResizer
from shared_libs.data_processing.video_components.cleaners.frame_rate_adjuster import FrameRateAdjuster
from shared_libs.data_processing.video_components.cleaners.video_noise_reducer import VideoNoiseReducer
from shared_libs.data_processing.video_components.cleaners.motion_stabilizer import MotionStabilizer

logger = logging.getLogger(__name__)

class VideoCleanerFactory:
    """
    A factory class for creating different types of video cleaning components.

    This class centralizes the creation logic, allowing for a config-driven
    approach to building video processing pipelines.
    """
    _CLEANER_MAP: Dict[str, Type[BaseVideoCleaner]] = {
        "video_resizer": VideoFrameResizer,
        "frame_rate_adjuster": FrameRateAdjuster,
        "video_noise_reducer": VideoNoiseReducer,
        "motion_stabilizer": MotionStabilizer,
        # Add new video cleaners here
    }

    @classmethod
    def create(cls, cleaner_type: str, config: Optional[Dict[str, Any]] = None) -> BaseVideoCleaner:
        """
        Creates and returns a video cleaner instance based on its type.

        Args:
            cleaner_type (str): The type of cleaner to create (e.g., "video_resizer").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseVideoCleaner: An instance of the requested cleaner.

        Raises:
            ValueError: If the specified cleaner_type is not supported.
            RuntimeError: If instantiation fails due to bad parameters.
        """
        config = config or {}
        cleaner_type_lower = cleaner_type.lower()
        cleaner_cls = cls._CLEANER_MAP.get(cleaner_type_lower)
        
        if not cleaner_cls:
            supported_cleaners = ", ".join(cls._CLEANER_MAP.keys())
            logger.error(f"Unsupported video cleaner type: '{cleaner_type}'. Supported types are: {supported_cleaners}")
            raise ValueError(f"Unsupported video cleaner type: '{cleaner_type}'. Supported types are: {supported_cleaners}")

        logger.info(f"Creating instance of {cleaner_cls.__name__}...")
        try:
            # Hardening: Wrap instantiation for robust error handling
            return cleaner_cls(**config)
        except Exception as e:
            logger.error(f"Factory failed to instantiate {cleaner_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{cleaner_type}': {e}")