# shared_libs/data_processing/image_components/augmenters/image_augmenter_factory.py

import logging
from typing import Dict, Any, Type, Optional, Union
# Import contracts
from shared_libs.data_processing._base.base_augmenter import BaseAugmenter
from shared_libs.data_processing.configs.preprocessing_config_schema import AugmentationConfig
# Import Orchestrator (for composition)
from shared_libs.data_processing.image_components.augmenters.image_augmenter_orchestrator import ImageAugmenterOrchestrator
# Import Atomic Augmenters (assuming they are placed correctly)
from shared_libs.data_processing.image_components.augmenters.atomic.flip_rotate import FlipRotate
from shared_libs.data_processing.image_components.augmenters.atomic.noise_injection import NoiseInjection
from shared_libs.data_processing.image_components.augmenters.atomic.cutmix import CutMix
from shared_libs.data_processing.image_components.augmenters.atomic.mixup import Mixup

logger = logging.getLogger(__name__)

class ImageAugmenterFactory:
    """
    A factory class for creating different types of data augmenters and the Orchestrator.

    It centralizes the creation logic and provides a static method to construct 
    the policy-aware Orchestrator from a configuration.
    """
    _AUGMENTER_MAP: Dict[str, Type[BaseAugmenter]] = {
        "flip_rotate": FlipRotate,
        "noise_injection": NoiseInjection,
        "cutmix": CutMix,
        "mixup": Mixup,
        # Add new augmenters here
    }

    @classmethod
    def create(cls, augmenter_type: str, config: Optional[Dict[str, Any]] = None) -> BaseAugmenter:
        """
        Creates and returns a single data augmenter instance based on its type.
        
        Args:
            augmenter_type (str): The type of augmenter to create (e.g., "cutmix").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseAugmenter: An instance of the requested augmenter.

        Raises:
            ValueError: If the specified augmenter_type is not supported.
            RuntimeError: If instantiation fails due to bad parameters.
        """
        config = config or {}
        augmenter_type_lower = augmenter_type.lower()
        augmenter_cls = cls._AUGMENTER_MAP.get(augmenter_type_lower)
        
        if not augmenter_cls:
            supported_augmenters = ", ".join(cls._AUGMENTER_MAP.keys())
            logger.error(f"Unsupported augmenter type: '{augmenter_type}'. Supported types are: {supported_augmenters}")
            raise ValueError(f"Unsupported augmenter type: '{augmenter_type}'. Supported types are: {supported_augmenters}")

        logger.info(f"Creating instance of {augmenter_cls.__name__}...")
        try:
            return augmenter_cls(**config)
        except Exception as e:
            logger.error(f"Factory failed to instantiate {augmenter_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{augmenter_type}': {e}")


    @staticmethod
    def build_from_config(config: Union[Dict[str, Any], AugmentationConfig]) -> ImageAugmenterOrchestrator:
        """
        Constructs the policy-aware ImageAugmenterOrchestrator instance based on the master configuration.

        Args:
            config (Union[Dict[str, Any], AugmentationConfig]): The master configuration 
                                                               object (dict or Pydantic model) 
                                                               containing both steps and policy settings.

        Returns:
            ImageAugmenterOrchestrator: The initialized orchestrator.
        
        Raises:
            ValueError: If the configuration cannot be validated by Pydantic.
        """
        # Hardening: Validate and normalize the input configuration using the schema
        if not isinstance(config, AugmentationConfig):
            try:
                # Ensure the configuration adheres to the Pydantic schema (with Policy fields)
                validated_config = AugmentationConfig(**config)
            except Exception as e:
                logger.error(f"Augmentation Config Validation Failed in Factory: {e}")
                raise ValueError(f"Invalid Augmentation Configuration: {e}")
        else:
            validated_config = config

        # 1. Extract policy parameters safely from the validated configuration
        policy_type = validated_config.policy_mode
        n_select = validated_config.n_select
        
        # 2. Instantiate the Orchestrator
        # The Orchestrator expects the raw dictionary of the AugmentationConfig section to re-validate internally,
        # but for simplicity and decoupling, we pass the validated object's dict representation.
        orchestrator = ImageAugmenterOrchestrator(
            config=validated_config.dict(), # Pass the validated dict representation
            policy_type=policy_type,
            n_select=n_select
        )
        
        return orchestrator