import logging
from typing import Dict, Any, Type, Optional

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter
from shared_libs.data_processing.augmenters.atomic.flip_rotate import FlipRotate
from shared_libs.data_processing.augmenters.atomic.noise_injection import NoiseInjection
from shared_libs.data_processing.augmenters.atomic.cutmix import CutMix
from shared_libs.data_processing.augmenters.atomic.mixup import Mixup

logger = logging.getLogger(__name__)

class AugmenterFactory:
    """
    A factory class for creating different types of data augmenters.

    This class centralizes the creation logic, allowing for a config-driven
    approach to building image augmentation pipelines.
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
        Creates and returns a data augmenter instance based on its type.

        Args:
            augmenter_type (str): The type of augmenter to create (e.g., "cutmix", "flip_rotate").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the augmenter's constructor.

        Returns:
            BaseAugmenter: An instance of the requested augmenter.

        Raises:
            ValueError: If the specified augmenter_type is not supported.
        """
        config = config or {}
        augmenter_cls = cls._AUGMENTER_MAP.get(augmenter_type.lower())
        
        if not augmenter_cls:
            supported_augmenters = ", ".join(cls._AUGMENTER_MAP.keys())
            logger.error(f"Unsupported augmenter type: '{augmenter_type}'. Supported types are: {supported_augmenters}")
            raise ValueError(f"Unsupported augmenter type: '{augmenter_type}'. Supported types are: {supported_augmenters}")

        logger.info(f"Creating instance of {augmenter_cls.__name__}...")
        return augmenter_cls(**config)