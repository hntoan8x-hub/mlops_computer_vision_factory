# shared_libs/data_labeling/auto_annotation/classification_proposal.py

import logging
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from torch import Tensor
from pydantic import ValidationError

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
# Import Schema Trusted Labels
from ...configs.label_schema import ClassificationLabel

logger = logging.getLogger(__name__)

class ClassificationProposalAnnotator(BaseAutoAnnotator):
    """
    Specialized Annotator for Image Classification: Assigns a single class label to the entire image.
    
    This annotator uses models like CLIP, ViT, or EfficientNet to predict the class 
    and validates the output against the strict ClassificationLabel schema.
    """
    
    def _run_inference(self, image_data: np.ndarray) -> Tuple[str, float]:
        """
        Simulates running a classification model and returns the predicted class and confidence.

        Args:
            image_data: The input image as a NumPy array (H, W, C).

        Returns:
            Tuple[str, float]: (Predicted class name, Confidence score).
        """
        # Simulate prediction results based on image properties (in a real scenario, this calls self.model.predict)
        
        # Example: Predict "cat" or "dog" based on mean pixel value
        if np.mean(image_data) > 128:
            predicted_class = "cat"
            confidence = 0.92
        else:
            predicted_class = "dog"
            confidence = 0.85
            
        return predicted_class, confidence

    def _normalize_output(self, raw_prediction: Tuple[str, float], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Applies confidence thresholding and validates the result against the Pydantic ClassificationLabel schema.
        
        Args:
            raw_prediction: The raw output from the classification model (class, confidence).
            metadata: Contextual info, must contain 'image_path'.

        Returns:
            List[StandardLabel]: A list containing one validated ClassificationLabel object, or an empty list if confidence is too low or validation fails.
        """
        predicted_class, confidence = raw_prediction
        image_path: str = metadata.get("image_path", "unknown")
        
        # 1. Apply confidence thresholding (Primary Filter)
        if confidence < self.min_confidence:
            logger.warning(f"[{image_path}] Skipping auto-label: Confidence ({confidence:.2f}) is below threshold ({self.min_confidence:.2f}).")
            return [] 

        # 2. Create and Validate ClassificationLabel (Trusted Label)
        try:
            # ClassificationLabel requires image_path and label (class name)
            label_obj = ClassificationLabel(
                image_path=image_path,
                label=predicted_class,
                # Confidence is an important metadata piece, often added directly to the output object
                # We can dump the Pydantic model and inject confidence for downstream tracking
            )
            
            # Convert to dictionary and add confidence for compatibility if needed, 
            # otherwise return the Pydantic object directly. 
            # Here we return the Pydantic object as StandardLabel (Union).
            
            # NOTE: We can inject confidence as a tracking metric here if the schema allows extra fields 
            # (though BaseLabel uses extra="forbid" for strictness, we return the object).
            
            return [label_obj]
            
        except ValidationError as e:
            # Hardening: Catch Pydantic validation failure (e.g., predicted_class is an empty string)
            logger.error(f"[{image_path}] Invalid ClassificationLabel filtered: {e}. Raw Class: {predicted_class}")
            return []
            
        except Exception as e:
            logger.error(f"[{image_path}] Unexpected error during ClassificationLabel creation: {e}")
            return []