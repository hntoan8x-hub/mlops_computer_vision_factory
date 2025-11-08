# shared_libs/data_processing/image_components/augmenters/atomic/flip_rotate.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class FlipRotate(BaseAugmenter):
    """
    Applies random horizontal/vertical flips and rotations to images.

    Rotation intensity can be controlled via the magnitude parameter for 
    policy-based augmentation (RandAugment).
    """
    def __init__(self, horizontal_flip: float = 0.5, vertical_flip: float = 0.0, rotate_angle: Optional[int] = 90):
        """
        Initializes the FlipRotate augmenter.

        Args:
            horizontal_flip (float): Probability of applying a horizontal flip (0.0 to 1.0).
            vertical_flip (float): Probability of applying a vertical flip (0.0 to 1.0).
            rotate_angle (Optional[int]): The maximum absolute angle for random rotation (e.g., 90 for -90 to 90).
        
        Raises:
            ValueError: If flip probabilities are outside the [0.0, 1.0] range.
        """
        if not 0.0 <= horizontal_flip <= 1.0 or not 0.0 <= vertical_flip <= 1.0:
            raise ValueError("Flip probabilities must be between 0.0 and 1.0.")
            
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.max_rotate_angle = rotate_angle if rotate_angle is not None else 90 
        logger.info("Initialized FlipRotate augmenter.")

    def transform(self, image: ImageData, magnitude: float = 1.0, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies flip and rotate augmentations, controlled by magnitude.

        Args:
            image (ImageData): The input image(s).
            magnitude (float): The intensity of the transformation (0.0 to 1.0), 
                               used to scale the rotation angle.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ImageData: The augmented image(s).
        
        Raises:
            TypeError: If the input is not a NumPy array or a list of NumPy arrays.
        """
        if isinstance(image, np.ndarray):
            return self._transform_single_image(image, magnitude)
        elif isinstance(image, list):
            return [self._transform_single_image(img, magnitude) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _transform_single_image(self, img: np.ndarray, magnitude: float) -> np.ndarray:
        """
        Helper method to apply flip and rotate to a single image.
        """
        aug_img = img.copy()
        
        if img.ndim < 2:
             raise ValueError("Input image must have at least 2 dimensions (H x W).")
             
        # Horizontal Flip
        if np.random.rand() < self.horizontal_flip:
            aug_img = cv2.flip(aug_img, 1)

        # Vertical Flip
        if np.random.rand() < self.vertical_flip:
            aug_img = cv2.flip(aug_img, 0)
        
        # Rotation - Hardening Magnitude Control
        if self.max_rotate_angle > 0 and magnitude > 0:
            # Scale the max angle by the magnitude (e.g., 90 * 0.5 = 45 degrees max)
            max_scaled_angle = int(self.max_rotate_angle * magnitude)
            
            rows, cols = aug_img.shape[:2]
            
            # Rotation angle is a random value within the scaled range
            angle = np.random.randint(-max_scaled_angle, max_scaled_angle + 1)
            
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            aug_img = cv2.warpAffine(aug_img, M, (cols, rows))
            
        return aug_img