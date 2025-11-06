# shared_libs/data_labeling/auto_annotation/factory.py

import logging
from typing import Dict, Any, Type
from .base_auto_annotator import BaseAutoAnnotator
from .detection_proposal import DetectionProposalAnnotator
from .classification_proposal import ClassificationProposalAnnotator 
from .segmentation_proposal import SegmentationProposalAnnotator 
from .ocr_proposal import OCRProposalAnnotator 
# IMPORT MỚI: EmbeddingProposalAnnotator
from .embedding_proposal import EmbeddingProposalAnnotator 

logger = logging.getLogger(__name__)

class AutoAnnotatorFactory:
    """
    Factory class quản lý việc tạo ra các Auto Proposal Annotator chuyên biệt.
    """
    
    AUTO_ANNOTATOR_MAPPING: Dict[str, Type[BaseAutoAnnotator]] = {
        "detection": DetectionProposalAnnotator,
        "classification": ClassificationProposalAnnotator,
        "segmentation": SegmentationProposalAnnotator,
        "ocr": OCRProposalAnnotator,
        "embedding": EmbeddingProposalAnnotator, # Đã thêm
    }

    @staticmethod
    def get_annotator(domain_type: str, config: Dict[str, Any]) -> BaseAutoAnnotator:
        """
        Tạo và trả về một Auto Proposal Annotator instance.
        """
        domain_type = domain_type.lower()
        if domain_type not in AutoAnnotatorFactory.AUTO_ANNOTATOR_MAPPING:
            raise ValueError(f"Unsupported auto-annotation domain: {domain_type}")

        AnnotatorClass = AutoAnnotatorFactory.AUTO_ANNOTATOR_MAPPING[domain_type]
        
        try:
            return AnnotatorClass(config)
        except Exception as e:
            logger.error(f"Failed to instantiate {domain_type} Auto Annotator: {e}")
            raise RuntimeError(f"Auto Annotator creation failed: {e}")