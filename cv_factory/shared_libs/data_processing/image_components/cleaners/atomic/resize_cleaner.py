# shared_libs/data_processing/image_components/cleaners/atomic/resize_cleaner.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner, ImageData

logger = logging.getLogger(__name__)

class ResizeCleaner(BaseImageCleaner):
    """
    Resizes images to a specified width and height.

    The component ensures that the input image is of a type compatible with 
    OpenCV's resizing functions.
    """
    def __init__(self, width: int, height: int, interpolation: int = cv2.INTER_AREA):
        """
        Initializes the ResizeCleaner.

        Args:
            width (int): The target width of the image.
            height (int): The target height of the image.
            interpolation (int, optional): The interpolation method for resizing.
                                 Defaults to cv2.INTER_AREA, suitable for shrinking images.

        Raises:
            ValueError: If width or height are not positive integers.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        self.width = width
        self.height = height
        self.interpolation = interpolation
        logger.info(f"Initialized ResizeCleaner to {self.width}x{self.height} with interpolation {self.interpolation}.")

    def transform(self, image: ImageData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Resizes a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            metadata (Optional[Dict[str, Any]]): Metadata of the input image (unused but kept for interface consistency).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ImageData: The resized image(s).

        Raises:
            TypeError: If the input is not a NumPy array or a list of NumPy arrays.
        """
        if isinstance(image, np.ndarray):
            return self._resize_single_image(image)
        elif isinstance(image, list):
            return [self._resize_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _resize_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to resize a single image, with type hardening.
        """
        if img.ndim < 2:
            logger.error(f"Image dimension is too low: {img.ndim}.")
            raise ValueError(f"Image must have at least 2 dimensions (H x W). Found {img.ndim}.")
            
        original_dtype = img.dtype
        if original_dtype not in (np.uint8, np.float32):
            logger.warning(
                f"Resizing image with non-standard dtype {original_dtype}. "
                "Converting to np.float32 for safer processing."
            )
            img = img.astype(np.float32)

        try:
            resized_img = cv2.resize(img, (self.width, self.height), interpolation=self.interpolation)
            # The output type should be handled by the next cleaner (e.g., NormalizeCleaner expects float)
            return resized_img
        except Exception as e:
            logger.error(f"Failed to resize image (H={img.shape[0]}, W={img.shape[1]}). Error: {e}")
            raise