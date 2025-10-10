import numpy as np
import logging
from typing import Dict, Any, Union, List

from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner, ImageData

logger = logging.getLogger(__name__)

class NormalizeCleaner(BaseImageCleaner):
    """
    Normalizes image pixel values.

    The normalization formula is: normalized_pixel = (pixel - mean) / std.
    """
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        """
        Initializes the NormalizeCleaner.

        Args:
            mean (Union[float, List[float]]): The mean value(s) for normalization.
                                              Can be a single float or a list (e.g., for RGB channels).
            std (Union[float, List[float]]): The standard deviation value(s) for normalization.
                                             Can be a single float or a list.
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        logger.info(f"Initialized NormalizeCleaner with mean={self.mean} and std={self.std}.")

    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Normalizes a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The normalized image(s).
        """
        if isinstance(image, np.ndarray):
            return self._normalize_single_image(image)
        elif isinstance(image, list):
            return [self._normalize_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _normalize_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to normalize a single image.
        """
        try:
            # Convert image to float32 before normalization
            img = img.astype(np.float32)
            normalized_img = (img - self.mean) / self.std
            return normalized_img
        except Exception as e:
            logger.error(f"Failed to normalize image. Error: {e}")
            raise