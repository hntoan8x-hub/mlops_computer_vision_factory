# shared_libs/data_labeling/implementations/detection_labeler.py (Hardened)

import json
from torch import tensor, float32, long
from typing import Dict, Any, List, Union, Tuple, Literal
import logging

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import DetectionLabel
from ...data_labeling.configs.labeler_config_schema import DetectionLabelerConfig

# IMPORT CÁC FACTORY
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory
from ..semi_annotation.semi_annotator_factory import SemiAnnotatorFactory

logger = logging.getLogger(__name__)

class DetectionLabeler(BaseLabeler):
    """
    Concrete Labeler for Object Detection. 
    
    Orchestrates the three modes: Manual Parsing, Auto Annotation, and Semi-Annotation Refinement.
    
    Attributes:
        annotation_mode (Literal): Current mode of operation ("manual", "auto", or "semi").
        auto_annotator (AutoAnnotatorFactory): Instance for generating proposals.
        semi_annotator (SemiAnnotatorFactory): Instance for refinement/selection.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.class_name_to_id: Dict[str, int] = {"__background__": 0, "person": 1, "car": 2} # Hardening: Add background class
        self.annotation_mode: Literal["manual", "auto", "semi"] = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        self.config_params: DetectionLabelerConfig = self.validated_config.params

        self.auto_annotator = self._initialize_auto_annotator()
        self.semi_annotator = self._initialize_semi_annotator()


    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., ProposalGenerator) if needed."""
        if self.annotation_mode in ["auto", "semi"]:
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "detection") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def _initialize_semi_annotator(self):
        """Initializes Semi Annotator (e.g., RefinementAnnotator) if needed."""
        if self.annotation_mode == "semi":
            semi_config = self.validated_config.raw_config.get("semi_annotation", {})
            if semi_config:
                method_type = semi_config.get("method_type", "refinement")
                return SemiAnnotatorFactory.get_annotator(method_type, semi_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads raw labels or image metadata based on the annotation mode.
        """
        source_uri = self.config_params.label_source_uri
        
        # 1. Load Raw Data/Metadata
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # 2a. MANUAL MODE: Use Manual Annotator (Parser)
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="detection", 
                    config=self.validated_config.model_dump() # Pass the full validated config
                )
                validated_labels_pydantic: List[DetectionLabel] = parser.parse(raw_data)
                
                # Hardening: Convert Pydantic object to dictionary for DataLoader
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Detection manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode in ["auto", "semi"]:
            # 2b. AUTO/SEMI MODE: Raw data is image metadata.
            
            # Hardening: Ensure raw_data is converted to a List[Dict] structure
            if isinstance(raw_data, dict) and "images" in raw_data:
                 final_labels = raw_data["images"]
            elif isinstance(raw_data, list):
                 final_labels = raw_data
            else:
                 raise TypeError("Auto/Semi mode expects List[Dict] or Dict with 'images' key.")
            
            logger.info(f"Loaded {len(final_labels)} samples for {self.annotation_mode} annotation. Actual annotation runs per-sample.")
            
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")


        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Performs validation of a loaded label sample.
        """
        objects = sample.get("objects", [])
        if not objects:
            return False
            
        # Hardening: Re-validate BBox semantic rules (already done by Pydantic, but good runtime check)
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid BBox bounds found in sample: {sample.get('image_path')}")
                return False
            if obj['class_name'] not in self.class_name_to_id:
                 logger.warning(f"Unknown class name '{obj['class_name']}' found. Skipping sample.")
                 return False
                 
        return True

    def convert_to_tensor(self, sample: Dict[str, Any]) -> Tuple[tensor, tensor]:
        """
        Converts Bounding Boxes and Class IDs into PyTorch Tensors.
        """
        objects: List[Dict[str, Any]] = sample.get("objects", [])
        if not objects:
            return tensor([]).float(), tensor([]).long()

        bboxes: List[Tuple[float, float, float, float]] = [obj["bbox"] for obj in objects]
        
        # Hardening: Map class name to ID, defaulting to 0 (__background__) if missing
        class_ids: List[int] = [self.class_name_to_id.get(obj["class_name"], 0) for obj in objects] 

        bbox_tensor = tensor(bboxes, dtype=float32)
        class_id_tensor = tensor(class_ids, dtype=long)
        
        return bbox_tensor, class_id_tensor