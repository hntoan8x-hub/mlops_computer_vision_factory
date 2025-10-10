# domain_models/medical_imaging/utils/validation_utils.py (RESTRUCTURED AND FINALIZED)

import logging
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def is_valid_medical_image(image: np.ndarray, min_shape: Tuple[int, int] = (100, 100), max_value: int = 4095) -> bool:
    """
    Checks if a medical image is valid based on domain-specific rules.
    (Logic moved from medical_rules_utils.py)
    
    Args:
        image (np.ndarray): The input image.
        min_shape (tuple): The minimum required shape (H, W).
        max_value (int): The maximum expected pixel value (e.g., 12-bit max).

    Returns:
        bool: True if the image is valid based on clinical rules, False otherwise.
    """
    if not isinstance(image, np.ndarray):
        return False
        
    # Check minimum size rule
    if image.ndim < 2 or image.shape[0] < min_shape[0] or image.shape[1] < min_shape[1]:
        logger.warning(f"Image shape {image.shape} is smaller than minimum required {min_shape} or dimension too low.")
        return False
        
    # Check maximum pixel value (important for medical grayscale images)
    if np.max(image) > max_value:
        logger.warning(f"Image pixel value exceeds max allowed value of {max_value}.")
        return False
        
    return True