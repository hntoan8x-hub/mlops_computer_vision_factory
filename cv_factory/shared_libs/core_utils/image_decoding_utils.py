# shared_libs/core_utils/image_decoding_utils.py (NEW FILE)

import numpy as np
import base64
import io
from PIL import Image
from typing import Optional

def decode_base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Converts a base64 encoded string back into an RGB NumPy array.
    
    This is a core utility function, decoupled from the CVPredictor logic.
    """
    if not base64_string:
        raise ValueError("Input base64 string cannot be empty.")
        
    try:
        # 1. Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)
        # 2. Open image from bytes stream using PIL
        img_stream = io.BytesIO(img_bytes)
        with Image.open(img_stream) as img:
            # 3. Convert to RGB and then to NumPy array
            return np.array(img.convert("RGB"))
    except Exception as e:
        # Reroute the exception to a more informative message
        raise ValueError(f"Base64 decoding or PIL conversion failed: {type(e).__name__} - check input format.")