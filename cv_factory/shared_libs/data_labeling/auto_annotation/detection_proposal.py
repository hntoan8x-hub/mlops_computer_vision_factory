# shared_libs/data_labeling/auto_annotation/detection_proposal.py

import numpy as np
import os
from typing import List, Dict, Any, Tuple, Union
import logging

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
# Import các Schema Trusted Labels đã định nghĩa
from ...configs.label_schema import DetectionLabel, DetectionObject, BBoxNormalized
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class DetectionProposalAnnotator(BaseAutoAnnotator):
    """
    Specialized Annotator for Object Detection: Generates Bounding Boxes and Class IDs.
    
    This annotator typically uses models like YOLO or FasterRCNN to produce raw 
    predictions and normalizes them into the strict DetectionLabel schema.
    """
    
    def _run_inference(self, image_data: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Simulates running a Detection model (e.g., YOLO/FasterRCNN) and returns a list: 
        [(bbox_xyxy_pixel, class_name, confidence), ...].
        
        Args:
            image_data: The input image as a NumPy array (H, W, C).

        Returns:
            List[Tuple[Tuple[int, int, int, int], str, float]]: List of raw predictions 
                (pixel BBox, class name, confidence).
        """
        H, W, _ = image_data.shape
        
        # Simulate predictions: BBox in pixel coordinates, Class name, Confidence
        predictions = [
            # High confidence, valid object
            ((int(W*0.1), int(H*0.1), int(W*0.5), int(H*0.5)), "person", 0.95),
            # Valid object, moderate confidence
            ((int(W*0.6), int(H*0.6), int(W*0.8), int(H*0.8)), "car", 0.80),
            # Low confidence (will be filtered by min_confidence)
            ((int(W*0.2), int(H*0.2), int(W*0.3), int(H*0.3)), "noise", 0.50), 
            # Invalid BBox (x_min >= x_max) - will fail Pydantic validation
            ((int(W*0.9), int(H*0.1), int(W*0.8), int(H*0.5)), "invalid_box", 0.90), 
        ]
        return predictions

    def _normalize_output(self, raw_prediction: List[Tuple[Tuple, str, float]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Normalizes pixel BBoxes to [0, 1], applies confidence thresholding, 
        and validates against the Pydantic DetectionObject schema.
        
        Args:
            raw_prediction: The raw output from the detection model.
            metadata: Contextual info, must contain 'image_path' and 'image_data'.

        Returns:
            List[StandardLabel]: A list containing one validated DetectionLabel object (or empty list if no objects pass).
        """
        suggested_objects: List[DetectionObject] = []
        image_path: str = metadata.get("image_path", "unknown")
        
        # Hardening: Get image dimensions from metadata for normalization
        image_data = metadata.get("image_data")
        if image_data is None:
            logger.error(f"Image data is missing in metadata for normalization of {image_path}.")
            return []
            
        img_h, img_w, _ = image_data.shape 

        for bbox_raw, class_name, confidence in raw_prediction:
            if confidence >= self.min_confidence:
                
                # 1. Normalize Bounding Box to [0, 1]
                x_min, y_min, x_max, y_max = bbox_raw
                try:
                    bbox_normalized: BBoxNormalized = (
                        float(x_min) / img_w,
                        float(y_min) / img_h,
                        float(x_max) / img_w,
                        float(y_max) / img_h
                    )
                except ZeroDivisionError:
                    logger.error(f"Cannot normalize BBox: Image dimensions are zero for {image_path}.")
                    continue

                # 2. Create and Validate DetectionObject (Trusted Label)
                try:
                    obj = DetectionObject(
                        bbox=bbox_normalized, 
                        class_name=class_name,
                        confidence=confidence
                    )
                    suggested_objects.append(obj)
                except ValidationError as e:
                    # Hardening: Catch Pydantic validation failure (e.g., BBox out of [0, 1], x_max <= x_min)
                    logger.warning(f"[{image_path}] Invalid Detection Object filtered: {e}. Raw BBox: {bbox_raw}")
                except Exception as e:
                    logger.error(f"Unexpected error during DetectionObject creation: {e}")


        # 3. Create the final DetectionLabel (enforcing not empty objects list)
        if suggested_objects:
            try:
                # DetectionLabel validation ensures image_path is valid and objects list is not empty
                final_label = DetectionLabel(image_path=image_path, objects=suggested_objects)
                return [final_label]
            except ValidationError as e:
                # This should only happen if image_path is invalid, or suggested_objects became empty later (unlikely here)
                logger.error(f"Failed to create valid DetectionLabel for {image_path}: {e}")
                return []
        else:
            # All proposals filtered out by confidence or Pydantic Validation
            return []