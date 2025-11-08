# shared_libs/data_labeling/auto_annotation/embedding_proposal.py

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from torch import Tensor
import os
from pydantic import ValidationError

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
# Import Schema Trusted Labels
from ...configs.label_schema import EmbeddingLabel

logger = logging.getLogger(__name__)

# Define type for Feature Vector (NumPy array or List[float])
FeatureVector = Union[np.ndarray, List[float]]

class EmbeddingProposalAnnotator(BaseAutoAnnotator):
    """
    Specialized Annotator for Embedding Learning (e.g., Face Recognition, Image Retrieval).
    
    This annotator generates feature vectors and the target ID/cluster ID for a given image.
    It enforces the dimensionality required by the EmbeddingLabel schema.
    """
    
    # Define the expected feature vector dimension
    EMBEDDING_DIM = 512 

    def _run_inference(self, image_data: np.ndarray) -> Tuple[FeatureVector, str]:
        """
        Simulates running the Embedding model (e.g., ResNet, CLIP) and returns: 
        (feature vector, nearest entity/cluster ID).
        
        Args:
            image_data: The input image as a NumPy array (H, W, C).

        Returns:
            Tuple[FeatureVector, str]: (512D feature vector, nearest entity ID).
        """
        # 1. Simulate Feature Vector generation
        # Hardening: Ensure the simulated vector matches the expected dimension
        feature_vector = np.random.rand(self.EMBEDDING_DIM).astype(np.float32)
        
        # 2. Simulate nearest ID lookup (e.g., from a face database)
        if np.mean(image_data) > 120:
            target_id = "person_A_v1"
        else:
            # Simulate a vector that leads to an 'unknown' or 'low quality' ID
            target_id = "unknown" 
            
        # Return as list for easier JSON/Pydantic serialization
        return feature_vector.tolist(), target_id

    def _normalize_output(self, raw_prediction: Tuple[FeatureVector, str], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Normalizes the raw prediction (vector/ID) into a validated EmbeddingLabel Pydantic object.
        
        Args:
            raw_prediction: The raw output (FeatureVector, target_id).
            metadata: Contextual info, must contain 'image_path'.

        Returns:
            List[StandardLabel]: A list containing one validated EmbeddingLabel object, or an empty list.
        """
        feature_vector, target_id = raw_prediction
        image_path: str = metadata.get("image_path", "unknown")
        
        # Hardening: Dimension check (if the model produced the wrong size)
        if len(feature_vector) != self.EMBEDDING_DIM:
             logger.error(f"[{image_path}] Vector dimension mismatch. Expected {self.EMBEDDING_DIM}, got {len(feature_vector)}. Skipping.")
             return []
        
        # 1. Create and Validate EmbeddingLabel (Trusted Label)
        try:
            label_obj = EmbeddingLabel(
                image_path=image_path,
                target_id=target_id,
                # Pydantic validation (in label_schema.py) will check:
                # - target_id is not empty
                # - vector is not empty
                vector=feature_vector 
            )
            
            # 2. Return as List[StandardLabel]
            return [label_obj]
            
        except ValidationError as e:
            # Hardening: Catch Pydantic validation failure (e.g., target_id is an empty string)
            logger.error(f"[{image_path}] Invalid EmbeddingLabel filtered: {e}. Raw Target ID: {target_id}")
            return []
            
        except Exception as e:
            logger.error(f"[{image_path}] Unexpected error during EmbeddingLabel creation: {e}")
            return []