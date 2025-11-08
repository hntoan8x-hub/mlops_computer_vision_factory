# shared_libs/data_labeling/manual_annotation/factory.py (Hardened)

import logging
from typing import Dict, Any, Type
from .base_manual_annotator import BaseManualAnnotator

# Import các Parser chuyên biệt
from .classification_parser import ClassificationParser
from .detection_parser import DetectionParser
from .segmentation_parser import SegmentationParser
from .ocr_parser import OCRParser
# (Nếu có EmbeddingParser, ta sẽ thêm vào đây)

logger = logging.getLogger(__name__)

class ManualAnnotatorFactory:
    """
    Factory class responsible for creating specific Manual Annotator (Parser) instances.
    
    It determines which Parser to instantiate based on the task_type provided in the configuration.
    This is essential for standardizing the ingestion of human-labeled data.
    """
    
    # Mapping task_type (domain) to Parser Class
    MANUAL_ANNOTATOR_MAPPING: Dict[str, Type[BaseManualAnnotator]] = {
        "classification": ClassificationParser,
        "detection": DetectionParser,
        "segmentation": SegmentationParser,
        "ocr": OCRParser,
    }

    @staticmethod
    def get_annotator(domain_type: str, config: Dict[str, Any]) -> BaseManualAnnotator:
        """
        Creates and returns a concrete Manual Annotator (Parser) instance.

        Args:
            domain_type: The CV task type ('classification', 'detection', ...).
            config: Detailed configuration for the Parser (e.g., format, columns). 
                    This config must conform to the corresponding LabelerConfigSchema.

        Returns:
            BaseManualAnnotator: An instance of the corresponding Parser class.

        Raises:
            ValueError: If the domain_type is not supported.
            RuntimeError: If the instantiation process (including Pydantic config validation) fails.
        """
        domain_type = domain_type.lower()
        if domain_type not in ManualAnnotatorFactory.MANUAL_ANNOTATOR_MAPPING:
            raise ValueError(f"Unsupported manual annotation domain: {domain_type}. Available types: {list(ManualAnnotatorFactory.MANUAL_ANNOTATOR_MAPPING.keys())}")

        AnnotatorClass = ManualAnnotatorFactory.MANUAL_ANNOTATOR_MAPPING[domain_type]
        
        try:
            # Instantiate the Parser, which immediately validates the config against its Pydantic Schema
            return AnnotatorClass(config)
        except Exception as e:
            # Hardening: Catch validation errors and initialization failures
            logger.error(f"Failed to instantiate {domain_type} Manual Annotator. Check configuration structure: {e}")
            raise RuntimeError(f"Manual Annotator creation failed for '{domain_type}': {e}")