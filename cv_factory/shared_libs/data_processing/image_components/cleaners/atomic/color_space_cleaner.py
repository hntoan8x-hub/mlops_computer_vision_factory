# shared_libs/data_processing/image_components/cleaners/atomic/color_space_cleaner.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Literal, Optional

from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner, ImageData

logger = logging.getLogger(__name__)

# Use Literal for type safety of color space conversions
ColorSpaceConversion = Literal["BGR2RGB", "RGB2BGR", "BGR2GRAY", "RGB2GRAY"]

class ColorSpaceCleaner(BaseImageCleaner):
    """
    Converts image color space (e.g., BGR to RGB).

    Exposes 'conversion_code' to the Policy Controller for conditional execution.
    """
    def __init__(self, conversion_code: ColorSpaceConversion):
        """
        Initializes the ColorSpaceCleaner.

        Args:
            conversion_code (ColorSpaceConversion): The code for color space conversion,
                                                    e.g., "BGR2RGB".
        """
        self.conversion_code = conversion_code # CRITICAL: Exposed for Controller to read
        self.cv2_code = self._get_cv2_code(conversion_code)
        logger.info(f"Initialized ColorSpaceCleaner with conversion code {self.conversion_code}.")

    def _get_cv2_code(self, code: str) -> int:
        """
        Maps a string code to an OpenCV conversion constant.
        """
        conversion_map = {
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "RGB2GRAY": cv2.COLOR_RGB2GRAY,
        }
        if code not in conversion_map:
            raise ValueError(f"Unsupported color space conversion code: {code}. Supported are {list(conversion_map.keys())}")
        return conversion_map[code]

    def transform(self, image: ImageData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies a color space conversion to a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            metadata (Optional[Dict[str, Any]]): Metadata of the input image (unused in atomic logic).
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The image(s) with converted color space.
        """
        if isinstance(image, np.ndarray):
            return self._convert_single_image(image)
        elif isinstance(image, list):
            return [self._convert_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _convert_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to convert a single image's color space, with input validation.
        """
        if img.ndim < 2 or img.ndim > 3:
            logger.error(f"Image dimension is {img.ndim}. Must be 2D or 3D for color conversion.")
            raise ValueError(f"Image must have 2 or 3 dimensions (H x W x C). Found {img.ndim}.")

        requires_color = "GRAY" not in self.conversion_code 
        is_grayscale_conversion = "GRAY" in self.conversion_code
        
        if img.ndim == 2 and not is_grayscale_conversion:
            raise ValueError(
                f"Cannot perform conversion '{self.conversion_code}' on a 2D (grayscale) image. "
                "Conversion requires a color image (3 channels)."
            )

        try:
            converted_img = cv2.cvtColor(img, self.cv2_code)
            return converted_img
        except Exception as e:
            logger.error(f"Failed to convert color space for image (Code: {self.conversion_code}). Error: {e}")
            raise