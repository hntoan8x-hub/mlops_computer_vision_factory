import numpy as np
import logging
from typing import Dict, Any, Union, List

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class NoiseInjection(BaseAugmenter):
    """
    Injects random Gaussian noise into an image.
    """
    def __init__(self, mean: float = 0.0, std_dev: float = 25.0, probability: float = 0.5):
        """
        Initializes the NoiseInjection augmenter.

        Args:
            mean (float): The mean of the Gaussian distribution.
            std_dev (float): The standard deviation of the Gaussian distribution.
            probability (float): Probability of applying noise to an image (0.0 to 1.0).
        """
        self.mean = mean
        self.std_dev = std_dev
        self.probability = probability
        logger.info(f"Initialized NoiseInjection with mean={self.mean}, std_dev={self.std_dev}.")

    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies noise injection to a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The augmented image(s).
        """
        if isinstance(image, np.ndarray):
            return self._inject_noise_single_image(image)
        elif isinstance(image, list):
            return [self._inject_noise_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _inject_noise_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to inject noise into a single image.
        """
        if np.random.rand() >= self.probability:
            return img
        
        try:
            row, col, ch = img.shape
            gauss = np.random.normal(self.mean, self.std_dev, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy_img = img + gauss
            
            # Clip pixel values to the valid range [0, 255]
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            return noisy_img
        except Exception as e:
            logger.error(f"Failed to inject noise into image. Error: {e}")
            return img