# shared_libs/data_processing/image_components/cleaners/image_cleaner_factory.py

import logging
from typing import Dict, Any, Type, Optional, Union

# Import Contracts
from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner
from shared_libs.data_processing.configs.preprocessing_config_schema import CleaningConfig # CRITICAL: Import CleaningConfig for validation
# Import Orchestrator (for composition)
from shared_libs.data_processing.image_components.cleaners.image_cleaner_orchestrator import ImageCleanerOrchestrator 
# Import Atomic Cleaners
from shared_libs.data_processing.image_components.cleaners.atomic.resize_cleaner import ResizeCleaner
from shared_libs.data_processing.image_components.cleaners.atomic.normalize_cleaner import NormalizeCleaner
from shared_libs.data_processing.image_components.cleaners.atomic.color_space_cleaner import ColorSpaceCleaner

logger = logging.getLogger(__name__)

class ImageCleanerFactory:
    """
    A factory class for creating different types of image cleaners and the Orchestrator.

    It centralizes the creation logic and provides a static method to construct 
    the policy-aware Orchestrator from a configuration.
    """
    _CLEANER_MAP: Dict[str, Type[BaseImageCleaner]] = {
        "resizer": ResizeCleaner,
        "normalizer": NormalizeCleaner,
        "color_space": ColorSpaceCleaner,
        # Add new cleaners here
    }

    @classmethod
    def create(cls, cleaner_type: str, config: Optional[Dict[str, Any]] = None) -> BaseImageCleaner:
        """
        Creates and returns a single image cleaner instance based on its type.
        
        Args:
            cleaner_type (str): The type of cleaner to create (e.g., "resizer").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseImageCleaner: An instance of the requested cleaner.

        Raises:
            ValueError: If the specified cleaner_type is not supported.
            RuntimeError: If instantiation fails due to bad parameters.
        """
        config = config or {}
        cleaner_type_lower = cleaner_type.lower()
        cleaner_cls = cls._CLEANER_MAP.get(cleaner_type_lower)
        
        if not cleaner_cls:
            supported_cleaners = ", ".join(cls._CLEANER_MAP.keys())
            logger.error(f"Unsupported cleaner type: '{cleaner_type}'. Supported types are: {supported_cleaners}")
            raise ValueError(f"Unsupported cleaner type: '{cleaner_type}'. Supported types are: {supported_cleaners}")

        logger.info(f"Creating instance of {cleaner_cls.__name__}...")
        try:
            return cleaner_cls(**config)
        except Exception as e:
            logger.error(f"Factory failed to instantiate {cleaner_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{cleaner_type}': {e}")


    @staticmethod
    def build_from_config(config: Union[Dict[str, Any], CleaningConfig]) -> ImageCleanerOrchestrator:
        """
        Constructs the policy-aware ImageCleanerOrchestrator instance based on the configuration.

        Args:
            config (Union[Dict[str, Any], CleaningConfig]): The master configuration 
                                                            object (dict or Pydantic model) 
                                                            containing the cleaning steps and policy settings.

        Returns:
            ImageCleanerOrchestrator: The initialized orchestrator.
        
        Raises:
            ValueError: If the configuration cannot be validated by Pydantic.
        """
        # Hardening: Validate and normalize the input configuration using the schema
        if not isinstance(config, CleaningConfig):
            try:
                # Ensure the configuration adheres to the Pydantic schema (including policy mode if added)
                validated_config = CleaningConfig(**config)
            except Exception as e:
                logger.error(f"Cleaning Config Validation Failed in Factory: {e}")
                raise ValueError(f"Invalid Cleaning Configuration: {e}")
        else:
            validated_config = config

        # 1. Extract Policy Parameters Safely (Assuming the config dict contains 'policy_mode')
        # NOTE: We assume 'policy_mode' is passed at the same level as 'steps' in the YAML/JSON.
        config_dict = validated_config.dict()
        policy_mode = config_dict.get('policy_mode', 'default') 
        
        # 2. Instantiate the Orchestrator
        # The Orchestrator will use the policy_mode extracted here.
        orchestrator = ImageCleanerOrchestrator(
            config=config_dict # Pass the validated dict representation
        )
        
        return orchestrator