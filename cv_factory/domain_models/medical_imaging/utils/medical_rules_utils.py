import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

def is_valid_medical_image(image: np.ndarray, min_shape: tuple = (100, 100), max_value: int = 4095) -> bool:
    """
    Checks if a medical image is valid based on domain-specific rules.

    Args:
        image (np.ndarray): The input image.
        min_shape (tuple): The minimum required shape (H, W).
        max_value (int): The maximum expected pixel value.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    if not isinstance(image, np.ndarray):
        return False
        
    if image.shape[0] < min_shape[0] or image.shape[1] < min_shape[1]:
        logger.warning(f"Image shape {image.shape} is smaller than minimum required {min_shape}.")
        return False
        
    if np.max(image) > max_value:
        logger.warning(f"Image pixel value exceeds max allowed value of {max_value}.")
        return False
        
    return True