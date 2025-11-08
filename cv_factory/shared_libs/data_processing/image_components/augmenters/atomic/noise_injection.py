# shared_libs/data_processing/image_components/augmenters/atomic/noise_injection.py

import numpy as np
import logging
from typing import Dict, Any, Union, List

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class NoiseInjection(BaseAugmenter):
    """
    Injects random Gaussian noise into an image or image batch.

    The intensity (Standard Deviation) is controlled by the magnitude parameter.
    """
    def __init__(self, mean: float = 0.0, std_dev: float = 25.0, probability: float = 0.5):
        """
        Initializes the NoiseInjection augmenter.

        Args:
            mean (float): The mean of the Gaussian distribution.
            std_dev (float): The base standard deviation (scaled by magnitude).
            probability (float): Probability of applying noise to an image (0.0 to 1.0).
        
        Raises:
            ValueError: If probability is outside the [0.0, 1.0] range.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0.")
            
        self.mean = mean
        self.base_std_dev = std_dev # Store the base value
        self.probability = probability
        logger.info(f"Initialized NoiseInjection with base std_dev={self.base_std_dev}.")

    def transform(self, image: ImageData, magnitude: float = 1.0, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies noise injection to a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            magnitude (float): The intensity of the transformation (0.0 to 1.0), 
                               used to scale the noise standard deviation.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ImageData: The augmented image(s).
        """
        if isinstance(image, np.ndarray):
            return self._inject_noise_single_image(image, magnitude)
        elif isinstance(image, list):
            return [self._inject_noise_single_image(img, magnitude) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _inject_noise_single_image(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """
        Helper method to inject noise into a single image.
        """
        if np.random.rand() >= self.probability or magnitude <= 0:
            return img
        
        try:
            # Hardening Magnitude Control: Scale std_dev
            scaled_std_dev = self.base_std_dev * magnitude
            
            # Convert to float for safe addition
            float_img = img.astype(np.float32)

            shape = img.shape
            # Generate Gaussian noise using the scaled standard deviation
            gauss = np.random.normal(self.mean, scaled_std_dev, shape).reshape(shape)
            noisy_img = float_img + gauss
            
            # Determine clipping range based on original dtype
            if img.dtype == np.uint8:
                clip_max = 255
            else:
                clip_max = np.finfo(img.dtype).max if np.issubdtype(img.dtype, np.floating) else np.iinfo(img.dtype).max
            
            # Clip pixel values and return to original dtype
            noisy_img = np.clip(noisy_img, 0, clip_max).astype(img.dtype)
            
            return noisy_img
        except Exception as e:
            logger.error(f"Failed to inject noise into image. Error: {e}")
            return img