# shared_libs/data_labeling/semi_annotation/semi_annotator_factory.py (Hardened)

import logging
from typing import Dict, Any, Type
from .base_semi_annotator import BaseSemiAnnotator
from .methods.refinement_annotator import RefinementAnnotator
from .methods.active_learning_selector import ActiveLearningSelector

logger = logging.getLogger(__name__)

class SemiAnnotatorFactory:
    """
    Factory class responsible for creating specialized Semi Annotator instances.
    
    It abstracts the creation of Active Learning selectors and label Refinement methods.
    """
    
    ANNOTATOR_MAPPING: Dict[str, Type[BaseSemiAnnotator]] = {
        "refinement": RefinementAnnotator,
        "active_learning": ActiveLearningSelector,
    }

    @staticmethod
    def get_annotator(method_type: str, config: Dict[str, Any]) -> BaseSemiAnnotator:
        """
        Creates and returns a concrete Semi Annotator instance.
        
        Args:
            method_type: The type of semi-annotation method ('refinement', 'active_learning', etc.).
            config: Configuration dictionary passed to the annotator's constructor.

        Returns:
            BaseSemiAnnotator: An instance of the requested annotator class.

        Raises:
            ValueError: If the method_type is not supported.
            RuntimeError: If instantiation fails.
        """
        method_type = method_type.lower()
        if method_type not in SemiAnnotatorFactory.ANNOTATOR_MAPPING:
            raise ValueError(
                f"Unsupported semi-annotation method: {method_type}. "
                f"Available methods: {list(SemiAnnotatorFactory.ANNOTATOR_MAPPING.keys())}"
            )

        AnnotatorClass = SemiAnnotatorFactory.ANNOTATOR_MAPPING[method_type]
        
        try:
            # Instantiate the annotator
            return AnnotatorClass(config)
        except Exception as e:
            logger.error(f"Failed to instantiate {method_type} Semi Annotator. Check config: {e}")
            raise RuntimeError(f"Semi Annotator creation failed: {e}")