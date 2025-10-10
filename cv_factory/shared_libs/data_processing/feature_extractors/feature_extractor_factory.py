import logging
from typing import Dict, Any, Type, Optional

from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor
from shared_libs.data_processing.feature_extractors.atomic.sift_extractor import SIFTExtractor
from shared_libs.data_processing.feature_extractors.atomic.orb_extractor import ORBExtractor
from shared_libs.data_processing.feature_extractors.atomic.hog_extractor import HOGExtractor

logger = logging.getLogger(__name__)

class FeatureExtractorFactory:
    """
    A factory class for creating different types of classical CV feature extractors.

    This class centralizes the creation logic, allowing for a config-driven
    approach to building feature extraction pipelines.
    """
    _EXTRACTOR_MAP: Dict[str, Type[BaseFeatureExtractor]] = {
        "sift": SIFTExtractor,
        "orb": ORBExtractor,
        "hog": HOGExtractor,
        # Add new classical feature extractors here
    }

    @classmethod
    def create(cls, extractor_type: str, config: Optional[Dict[str, Any]] = None) -> BaseFeatureExtractor:
        """
        Creates and returns a feature extractor instance based on its type.

        Args:
            extractor_type (str): The type of extractor to create (e.g., "sift", "orb", "hog").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the extractor's constructor.

        Returns:
            BaseFeatureExtractor: An instance of the requested extractor.

        Raises:
            ValueError: If the specified extractor_type is not supported.
        """
        config = config or {}
        extractor_cls = cls._EXTRACTOR_MAP.get(extractor_type.lower())
        
        if not extractor_cls:
            supported_extractors = ", ".join(cls._EXTRACTOR_MAP.keys())
            logger.error(f"Unsupported feature extractor type: '{extractor_type}'. Supported types are: {supported_extractors}")
            raise ValueError(f"Unsupported feature extractor type: '{extractor_type}'. Supported types are: {supported_extractors}")

        logger.info(f"Creating instance of {extractor_cls.__name__}...")
        return extractor_cls(**config)