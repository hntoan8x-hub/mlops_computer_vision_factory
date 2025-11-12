# shared_libs/data_labeling/auto_annotation/keypoint_proposal.py (NEW)

import numpy as np
import logging
from typing import List, Dict, Any, Union
from .base_auto_annotator import BaseAutoAnnotator
from ...configs.label_schema import KeypointLabel, KeypointObject, StandardLabel # Cần KeypointLabel

logger = logging.getLogger(__name__)

class KeypointProposalAnnotator(BaseAutoAnnotator):
    """
    Auto Annotator for Keypoint Estimation (e.g., Human Pose Estimation).
    
    The proposal output is the list of keypoint coordinates for detected objects.
    """

    def _run_inference(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Simulates running a Keypoint Estimation model.
        
        Returns:
            Dict[str, Any]: Raw prediction result (e.g., list of detected keypoint sets).
        """
        logger.info("Running simulated Keypoint Estimation inference.")
        # Giả định: Trả về 2 đối tượng (người)
        simulated_keypoints = [
            # Object 1 (17 keypoints * 2 coords)
            {'keypoints': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], 'score': 0.9}, 
            # Object 2
            {'keypoints': [[0.7, 0.8], [0.9, 0.95], [0.1, 0.15]], 'score': 0.8}
        ]
        return simulated_keypoints

    def _normalize_output(self, raw_prediction: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Normalizes raw keypoint predictions into validated KeypointLabel objects.
        """
        objects: List[KeypointObject] = []
        
        # Aggregate KeypointObjects that pass confidence
        for prediction in raw_prediction:
            score = prediction.get('score', 1.0)
            if score >= self.min_confidence:
                try:
                    objects.append(KeypointObject(
                        class_name=prediction.get('class_name', 'person'),
                        keypoints=prediction['keypoints'], # Normalized coords
                        confidence=score
                    ))
                except Exception as e:
                    logger.warning(f"Skipping invalid KeypointObject proposal: {e}")

        if objects:
            try:
                # Tạo KeypointLabel Pydantic object
                return [KeypointLabel(
                    image_path=metadata['image_path'],
                    objects=objects
                )]
            except Exception as e:
                logger.error(f"Failed to create KeypointLabel Pydantic object: {e}")
                return []
        
        return []