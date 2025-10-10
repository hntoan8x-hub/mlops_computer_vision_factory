import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List

from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner, ImageData

logger = logging.getLogger(__name__)

class ResizeCleaner(BaseImageCleaner):
    """
    Resizes images to a specified width and height.
    """
    def __init__(self, width: int, height: int, interpolation: int = cv2.INTER_AREA):
        """
        Initializes the ResizeCleaner.

        Args:
            width (int): The target width of the image.
            height (int): The target height of the image.
            interpolation (int): The interpolation method for resizing.
                                 Defaults to cv2.INTER_AREA, suitable for shrinking images.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        self.width = width
        self.height = height
        self.interpolation = interpolation
        logger.info(f"Initialized ResizeCleaner to {self.width}x{self.height} with interpolation {self.interpolation}.")

    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Resizes a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The resized image(s).
        """
        if isinstance(image, np.ndarray):
            return self._resize_single_image(image)
        elif isinstance(image, list):
            return [self._resize_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _resize_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to resize a single image.
        """
        try:
            resized_img = cv2.resize(img, (self.width, self.height), interpolation=self.interpolation)
            return resized_img
        except Exception as e:
            logger.error(f"Failed to resize image. Error: {e}")
            raise