import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class FlipRotate(BaseAugmenter):
    """
    Applies random horizontal/vertical flips and rotations to images.
    """
    def __init__(self, horizontal_flip: float = 0.5, vertical_flip: float = 0.0, rotate_angle: Optional[int] = None):
        """
        Initializes the FlipRotate augmenter.

        Args:
            horizontal_flip (float): Probability of applying a horizontal flip (0.0 to 1.0).
            vertical_flip (float): Probability of applying a vertical flip (0.0 to 1.0).
            rotate_angle (Optional[int]): The maximum absolute angle for random rotation (e.g., 90 for -90 to 90).
        """
        if not 0.0 <= horizontal_flip <= 1.0 or not 0.0 <= vertical_flip <= 1.0:
            raise ValueError("Flip probabilities must be between 0.0 and 1.0.")
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_angle = rotate_angle
        logger.info("Initialized FlipRotate augmenter.")

    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies flip and rotate augmentations to a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The augmented image(s).
        """
        if isinstance(image, np.ndarray):
            return self._transform_single_image(image)
        elif isinstance(image, list):
            return [self._transform_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _transform_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to apply flip and rotate to a single image.
        """
        aug_img = img.copy()
        
        # Horizontal Flip
        if np.random.rand() < self.horizontal_flip:
            aug_img = cv2.flip(aug_img, 1)

        # Vertical Flip
        if np.random.rand() < self.vertical_flip:
            aug_img = cv2.flip(aug_img, 0)
        
        # Rotation
        if self.rotate_angle is not None:
            rows, cols, _ = aug_img.shape
            angle = np.random.randint(-self.rotate_angle, self.rotate_angle + 1)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            aug_img = cv2.warpAffine(aug_img, M, (cols, rows))
            
        return aug_img