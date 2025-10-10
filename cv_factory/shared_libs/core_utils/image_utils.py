# cv_factory/shared_libs/core_utils/image_utils.py

import logging
import base64
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)

def decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Decodes a Base64-encoded image string into an RGB NumPy array.
    
    This function is a pure utility, used by CVPredictor to convert API/Kafka 
    payloads into a standardized image format for the preprocessing pipeline.

    Args:
        base64_string (str): The Base64 encoded image data (e.g., from a JSON payload).

    Returns:
        np.ndarray: The decoded image as a 3D NumPy array (H, W, C), in RGB format.

    Raises:
        ValueError: If Base64 decoding or image conversion fails.
    """
    if not base64_string:
        raise ValueError("Input base64 string cannot be empty.")
        
    try:
        # 1. Decode base64 string to raw binary bytes
        img_bytes = base64.b64decode(base64_string)
        
        # 2. Wrap the bytes in a stream for PIL
        img_stream = io.BytesIO(img_bytes)
        
        # 3. Open image using PIL
        with Image.open(img_stream) as img:
            # 4. Convert to RGB (standard for CV models) and then to NumPy array
            return np.array(img.convert("RGB"))
            
    except base64.binascii.Error as e:
        logger.error(f"Base64 decoding failed: Invalid padding or characters. {e}")
        raise ValueError(f"Invalid Base64 format: {e}") from e
    except IOError as e:
        logger.error(f"PIL failed to identify or open the image: {e}")
        raise ValueError(f"Invalid image data in Base64 payload: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during image decoding: {e}")
        raise ValueError(f"Image decoding failed: {e}") from e


# --- Example of another utility (Placeholder) ---

def resize_numpy_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """A pure utility function to resize a NumPy array (if needed outside the Orchestrator)."""
    # NOTE: Implementation would use OpenCV or PIL
    return image