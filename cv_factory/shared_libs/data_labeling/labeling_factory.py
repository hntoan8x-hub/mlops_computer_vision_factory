# shared_libs/data_labeling/labeling_factory.py (Hardened)

import logging
from typing import Dict, Any, Type

# Import Base Abstraction
from .base_labeler import BaseLabeler

# Import Config Schema để Validate Input
from .configs.labeler_config_schema import LabelerConfig

# Import Concrete Implementations
from .implementations.classification_labeler import ClassificationLabeler
from .implementations.detection_labeler import DetectionLabeler
from .implementations.segmentation_labeler import SegmentationLabeler
from .implementations.ocr_labeler import OCRLabeler
from .implementations.embedding_labeler import EmbeddingLabeler

logger = logging.getLogger(__name__)

class LabelingFactory:
    """
    Factory class responsible for creating instances of BaseLabeler based on task type.
    
    This acts as the main entry gate for the Data Labeling layer, validating the 
    top-level configuration before instantiation.
    """
    
    # Mapping task_type (string) to the specific Labeler Class
    LABELER_MAPPING: Dict[str, Type[BaseLabeler]] = {
        "classification": ClassificationLabeler,
        "detection": DetectionLabeler,
        "segmentation": SegmentationLabeler,
        "ocr": OCRLabeler,
        "embedding": EmbeddingLabeler,
    }

    @staticmethod
    def get_labeler(connector_id: str, raw_config: Dict[str, Any]) -> BaseLabeler:
        """
        Instantiates and returns a concrete Labeler.

        Args:
            connector_id: Unique ID for the Labeler.
            raw_config: Raw configuration dictionary (from file).

        Returns:
            BaseLabeler: An instance of the corresponding Labeler class.

        Raises:
            ValueError: If configuration is invalid or task type is unsupported.
            RuntimeError: If Labeler instantiation fails.
        """
        
        # 1. Validate configuration sơ bộ and extract task_type
        try:
            validated_config = LabelerConfig(**raw_config)
            task_type = validated_config.task_type
        except Exception as e:
            raise ValueError(f"Labeler configuration failed Pydantic validation for ID '{connector_id}': {e}")
            
        # 2. Select and Instantiate Labeler Class
        if task_type not in LabelingFactory.LABELER_MAPPING:
            raise ValueError(
                f"Unsupported labeler task type: '{task_type}'. "
                f"Available types are: {list(LabelingFactory.LABELER_MAPPING.keys())}"
            )
            
        LabelerClass = LabelingFactory.LABELER_MAPPING[task_type]
        
        try:
            # The high-level Labeler will handle auto-initialization of sub-factories inside __init__
            labeler = LabelerClass(
                connector_id=connector_id,
                config=raw_config
            )
            logger.info(f"[{connector_id}] Successfully created {task_type} Labeler.")
            return labeler
        except Exception as e:
            logger.error(f"[{connector_id}] Failed to instantiate {task_type} Labeler: {e}")
            raise RuntimeError(f"Labeler creation failed for type '{task_type}': {e}")