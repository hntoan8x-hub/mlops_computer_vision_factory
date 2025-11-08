# shared_libs/data_labeling/auto_annotation/ocr_proposal.py

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from torch import Tensor
import os
from pydantic import ValidationError

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
# Import Schema Trusted Labels
from ...configs.label_schema import OCRLabel, OCRToken, BBoxNormalized

logger = logging.getLogger(__name__)

class OCRProposalAnnotator(BaseAutoAnnotator):
    """
    Specialized Annotator for OCR/Text Extraction: Detects text regions (bbox) 
    and extracts content (full_text, tokens).
    
    This annotator typically uses models like Tesseract, EasyOCR, or PaddleOCR.
    """
    
    def _run_inference(self, image_data: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Simulates running an OCR model and returns a list of token predictions: 
        [(bbox_xyxy_pixel, text_content, confidence), ...].
        
        Args:
            image_data: The input image as a NumPy array (H, W, C).

        Returns:
            List[Tuple[Tuple[int, int, int, int], str, float]]: List of raw predictions 
                (pixel BBox, text content, confidence).
        """
        H, W, _ = image_data.shape
        
        # Simulate prediction results (words/tokens)
        predictions = [
            # Word 1: Hello (High confidence)
            ((int(W*0.1), int(H*0.1), int(W*0.3), int(H*0.2)), "Hello", 0.99),
            # Word 2: World! (Moderate confidence)
            ((int(W*0.3), int(H*0.1), int(W*0.5), int(H*0.2)), "World!", 0.95),
            # Text with low confidence (will be filtered)
            ((int(W*0.7), int(H*0.7), int(W*0.8), int(H*0.8)), "noise", 0.60),
            # Empty text content (will fail Pydantic validation)
            ((int(W*0.5), int(H*0.5), int(W*0.6), int(H*0.6)), "", 0.90), 
        ]
        return predictions

    def _normalize_output(self, raw_prediction: List[Tuple[Tuple, str, float]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Normalizes raw token predictions into Pydantic OCRToken objects, 
        and aggregates them into the final OCRLabel.
        
        Args:
            raw_prediction: The raw output from the OCR model.
            metadata: Contextual info, must contain 'image_path' and 'image_data'.

        Returns:
            List[StandardLabel]: A list containing one validated OCRLabel object, or an empty list.
        """
        image_path: str = metadata.get("image_path", "unknown")
        full_text_list: List[str] = []
        suggested_tokens: List[OCRToken] = []

        # Hardening: Get image dimensions from metadata for normalization
        image_data = metadata.get("image_data")
        if image_data is None:
            logger.error(f"Image data is missing in metadata for normalization of {image_path}.")
            return []
            
        img_h, img_w, _ = image_data.shape 
        
        for bbox_raw, text_content, confidence in raw_prediction:
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
                
                # 2. Create and Validate OCRToken (Trusted Label)
                try:
                    token = OCRToken(
                        text=text_content, 
                        bbox=bbox_normalized
                    )
                    suggested_tokens.append(token)
                    full_text_list.append(text_content)
                except ValidationError as e:
                    # Hardening: Catch Pydantic validation failure (e.g., empty text, invalid BBox)
                    logger.warning(f"[{image_path}] Invalid OCR Token filtered: {e}. Raw Content: '{text_content}'")
                except Exception as e:
                    logger.error(f"Unexpected error during OCRToken creation: {e}")

        # 3. Aggregate Full Text (join validated tokens)
        full_text = " ".join(full_text_list)
        
        # 4. Create the final OCRLabel
        if suggested_tokens:
            try:
                # OCRLabel validation ensures full_text is not empty and tokens list is not empty
                ocr_label = OCRLabel(
                    image_path=image_path,
                    full_text=full_text,
                    tokens=suggested_tokens
                )
                return [ocr_label]
            except ValidationError as e:
                # This ensures the aggregated label structure is valid
                logger.error(f"Failed to create valid aggregated OCRLabel for {image_path}: {e}")
                return []
        else:
            # All proposals filtered out by confidence or Pydantic Validation
            return []