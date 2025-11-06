# shared_libs/data_labeling/semi_annotation/semi_annotator_factory.py

import logging
from typing import Dict, Any, Type
from .base_semi_annotator import BaseSemiAnnotator
from .methods.refinement_annotator import RefinementAnnotator
from .methods.active_learning_selector import ActiveLearningSelector

logger = logging.getLogger(__name__)

class SemiAnnotatorFactory:
    """
    Factory class quản lý việc tạo ra các Semi Annotator chuyên biệt.
    """
    
    ANNOTATOR_MAPPING: Dict[str, Type[BaseSemiAnnotator]] = {
        "refinement": RefinementAnnotator,
        "active_learning": ActiveLearningSelector,
        # Thêm các phương pháp Semi-Annotation khác (ví dụ: weak supervision, consensus)
    }

    @staticmethod
    def get_annotator(method_type: str, config: Dict[str, Any]) -> BaseSemiAnnotator:
        """
        Tạo và trả về một Semi Annotator instance.
        """
        method_type = method_type.lower()
        if method_type not in SemiAnnotatorFactory.ANNOTATOR_MAPPING:
            raise ValueError(f"Unsupported semi-annotation method: {method_type}")

        AnnotatorClass = SemiAnnotatorFactory.ANNOTATOR_MAPPING[method_type]
        
        try:
            return AnnotatorClass(config)
        except Exception as e:
            logger.error(f"Failed to instantiate {method_type} Semi Annotator: {e}")
            raise RuntimeError(f"Semi Annotator creation failed: {e}")