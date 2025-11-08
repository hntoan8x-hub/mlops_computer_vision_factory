# shared_libs/data_labeling/auto_annotation/factory.py

import logging
from typing import Dict, Any, Type
from .base_auto_annotator import BaseAutoAnnotator
from .detection_proposal import DetectionProposalAnnotator
from .classification_proposal import ClassificationProposalAnnotator 
from .segmentation_proposal import SegmentationProposalAnnotator 
from .ocr_proposal import OCRProposalAnnotator 
from .embedding_proposal import EmbeddingProposalAnnotator 

logger = logging.getLogger(__name__)

class AutoAnnotatorFactory:
    """
    Factory class responsible for creating concrete instances of BaseAutoAnnotator 
    based on the specified domain/task type.
    
    This centralizes the instantiation logic for all auto-labeling proposal mechanisms.
    """
    
    AUTO_ANNOTATOR_MAPPING: Dict[str, Type[BaseAutoAnnotator]] = {
        "detection": DetectionProposalAnnotator,
        "classification": ClassificationProposalAnnotator,
        "segmentation": SegmentationProposalAnnotator,
        "ocr": OCRProposalAnnotator,
        "embedding": EmbeddingProposalAnnotator,
    }

    @staticmethod
    def get_annotator(domain_type: str, config: Dict[str, Any]) -> BaseAutoAnnotator:
        """
        Creates and returns a concrete Auto Proposal Annotator instance.

        Args:
            domain_type: The type of CV task/domain (e.g., 'detection', 'classification').
            config: Configuration dictionary passed to the annotator's constructor, 
                    typically containing model path and min_confidence.

        Returns:
            BaseAutoAnnotator: An instance of the requested concrete annotator.

        Raises:
            ValueError: If the domain_type is not supported.
            RuntimeError: If annotator instantiation (e.g., model loading) fails.
        """
        domain_type = domain_type.lower()
        
        if domain_type not in AutoAnnotatorFactory.AUTO_ANNOTATOR_MAPPING:
            raise ValueError(
                f"Unsupported auto-annotation domain: '{domain_type}'. "
                f"Available types are: {list(AutoAnnotatorFactory.AUTO_ANNOTATOR_MAPPING.keys())}"
            )

        AnnotatorClass = AutoAnnotatorFactory.AUTO_ANNOTATOR_MAPPING[domain_type]
        
        try:
            # Instantiate the annotator, which triggers model loading in __init__
            return AnnotatorClass(config)
        except Exception as e:
            # Hardening: Catch generic exceptions during instantiation (e.g., failed model load)
            logger.error(f"Failed to instantiate {domain_type} Auto Annotator. Check model path/type in config: {e}")
            raise RuntimeError(f"Auto Annotator creation failed for type '{domain_type}': {e}")