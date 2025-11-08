# shared_libs/data_labeling/auto_annotation/segmentation_proposal.py

import logging
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from torch import Tensor
from PIL import Image
from pydantic import ValidationError

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
# Import Schema Trusted Labels
from ...configs.label_schema import SegmentationLabel

logger = logging.getLogger(__name__)

class SegmentationProposalAnnotator(BaseAutoAnnotator):
    """
    Specialized Annotator for Image Segmentation: Generates pixel-wise mask labels.
    
    This annotator typically uses models like SAM (Segment Anything Model) or Mask R-CNN.
    The raw mask arrays are saved to a temporary file (e.g., PNG) and the path is 
    stored in the SegmentationLabel schema.
    """
    
    def _run_inference(self, image_data: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """
        Simulates running a Segmentation model and returns a list of predictions: 
        [(mask_array, class_name, confidence), ...].
        
        Args:
            image_data: The input image as a NumPy array (H, W, C).

        Returns:
            List[Tuple[np.ndarray, str, float]]: List of predicted masks (boolean/binary array), 
                class name, and confidence.
        """
        H, W, _ = image_data.shape
        
        # 1. Simulate Mask 1 (Center square)
        mask1 = np.zeros((H, W), dtype=bool)
        mask1[int(H*0.2):int(H*0.8), int(W*0.2):int(W*0.8)] = True
        
        # 2. Simulate Mask 2 (Top right area)
        mask2 = np.zeros((H, W), dtype=bool)
        mask2[:int(H*0.4), int(W*0.6):] = True
        
        # Simulate prediction results
        predictions = [
            (mask1, "main_object", 0.98),
            (mask2, "background_area", 0.70), 
            # Low confidence mask (will be filtered)
            (np.zeros((H, W), dtype=bool), "noise", 0.60), 
        ]
        return predictions

    def _normalize_output(self, raw_prediction: List[Tuple[np.ndarray, str, float]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Normalizes the binary mask array, saves it to a temporary file (PNG), 
        and validates the result against the Pydantic SegmentationLabel schema.
        
        Args:
            raw_prediction: The raw output from the segmentation model.
            metadata: Contextual info, must contain 'image_path'.

        Returns:
            List[StandardLabel]: A list of validated SegmentationLabel objects.
        """
        suggested_labels: List[SegmentationLabel] = []
        image_path: str = metadata.get("image_path", "unknown")
        
        # Hardening: Define and ensure existence of the temporary storage directory
        temp_mask_dir = self.config.get("temp_mask_dir", "/tmp/cvf_masks")
        try:
            os.makedirs(temp_mask_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create temporary mask directory {temp_mask_dir}: {e}")
            return []
        
        for idx, (mask_array, class_name, confidence) in enumerate(raw_prediction):
            if confidence >= self.min_confidence:
                
                # Hardening Check: Ensure mask is a valid size array
                if mask_array.ndim != 2 or mask_array.dtype not in [bool, np.uint8]:
                    logger.warning(f"[{image_path}] Skipping mask {idx}: Invalid array shape or dtype.")
                    continue
                    
                # 1. Normalize Mask: Convert binary mask (True/False) to 8-bit grayscale (0/255)
                mask_8bit = (mask_array * 255).astype(np.uint8)
                
                # 2. Save Mask to a temporary file
                # File Naming Convention: Ensure unique and traceable paths
                file_hash = hash(image_path) % 100000 
                mask_filename = f"{file_hash}_{idx}_{class_name}_{int(time.time())}.png"
                mask_save_path = os.path.join(temp_mask_dir, mask_filename)
                
                try:
                    # Save using PIL (mode 'L' for grayscale)
                    Image.fromarray(mask_8bit, 'L').save(mask_save_path)
                    
                    # 3. Create and Validate SegmentationLabel (Trusted Label)
                    label_obj = SegmentationLabel(
                        image_path=image_path,
                        mask_path=mask_save_path, # Path to the stored artifact
                        class_name=class_name
                    )
                    suggested_labels.append(label_obj)
                
                except ValidationError as e:
                    # Hardening: Catch Pydantic validation failure (e.g., mask_path is too short)
                    logger.error(f"[{image_path}] Invalid SegmentationLabel filtered: {e}")
                    # Remove the temp file if validation failed
                    if os.path.exists(mask_save_path):
                         os.remove(mask_save_path)
                         
                except Exception as e:
                    logger.error(f"Failed to save or create SegmentationLabel for {image_path}: {e}")
            
        return suggested_labels