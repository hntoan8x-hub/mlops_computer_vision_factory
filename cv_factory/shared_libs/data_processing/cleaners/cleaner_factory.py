import logging
from typing import Dict, Any, Type, Optional

from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner
from shared_libs.data_processing.cleaners.atomic.resize_cleaner import ResizeCleaner
from shared_libs.data_processing.cleaners.atomic.normalize_cleaner import NormalizeCleaner
from shared_libs.data_processing.cleaners.atomic.color_space_cleaner import ColorSpaceCleaner

logger = logging.getLogger(__name__)

class CleanerFactory:
    """
    A factory class for creating different types of image cleaners.

    This class centralizes the creation logic, allowing for a config-driven
    approach to building image processing pipelines.
    """
    _CLEANER_MAP: Dict[str, Type[BaseImageCleaner]] = {
        "resize": ResizeCleaner,
        "normalize": NormalizeCleaner,
        "color_space": ColorSpaceCleaner,
        # Add new cleaners here
    }

    @classmethod
    def create(cls, cleaner_type: str, config: Optional[Dict[str, Any]] = None) -> BaseImageCleaner:
        """
        Creates and returns an image cleaner instance based on its type.

        Args:
            cleaner_type (str): The type of cleaner to create (e.g., "resize", "normalize").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the cleaner's constructor.

        Returns:
            BaseImageCleaner: An instance of the requested cleaner.

        Raises:
            ValueError: If the specified cleaner_type is not supported.
        """
        config = config or {}
        cleaner_cls = cls._CLEANER_MAP.get(cleaner_type.lower())
        
        if not cleaner_cls:
            supported_cleaners = ", ".join(cls._CLEANER_MAP.keys())
            logger.error(f"Unsupported cleaner type: '{cleaner_type}'. Supported types are: {supported_cleaners}")
            raise ValueError(f"Unsupported cleaner type: '{cleaner_type}'. Supported types are: {supported_cleaners}")

        logger.info(f"Creating instance of {cleaner_cls.__name__}...")
        return cleaner_cls(**config)