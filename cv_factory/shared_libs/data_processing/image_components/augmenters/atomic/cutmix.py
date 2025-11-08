# shared_libs/data_processing/image_components/augmenters/atomic/cutmix.py
import numpy as np
import logging
from typing import Dict, Any, Union, List, Tuple

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class CutMix(BaseAugmenter):
    """
    Implements the CutMix data augmentation technique.

    Requires batches of images and their corresponding labels to mix.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the CutMix augmenter.
        """
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        self.alpha = alpha
        logger.info(f"Initialized CutMix with alpha={self.alpha}.")

    def transform(self, image_data: ImageData, magnitude: float = 1.0, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies the CutMix augmentation to a batch of images.

        Args:
            image_data (ImageData): A list of images (a batch) to be augmented.
            magnitude (float): The intensity of the transformation (0.0 to 1.0). 
                               Used for interface compliance.
            **kwargs: Must contain 'labels' (List or np.ndarray) for the corresponding batch.

        Returns:
            ImageData: The augmented batch of images.
        """
        if not isinstance(image_data, list) or len(image_data) < 2:
            raise ValueError("CutMix requires a list of at least two images (a batch).")

        batch_size = len(image_data)
        
        # HARDENING: Label Check
        if 'labels' not in kwargs:
            raise ValueError("CutMix augmentation requires a list or array of 'labels' in kwargs.")
        labels = kwargs['labels']
        if len(labels) != batch_size:
             raise ValueError(f"Label count ({len(labels)}) does not match batch size ({batch_size}).")
             
        # HARDENING: Ensure all images are of uniform shape
        try:
            height, width, channels = image_data[0].shape
        except (AttributeError, ValueError):
             raise TypeError("All input images must be NumPy arrays with HxWxC dimensions.")

        shuffle_indices = np.random.permutation(batch_size)
        shuffled_images = [image_data[i] for i in shuffle_indices]

        # Generate lambda from a Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Calculate the bounding box to be cut and pasted
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Get random coordinates for the bounding box
        x_center = np.random.randint(width)
        y_center = np.random.randint(height)
        
        x1 = np.clip(x_center - cut_w // 2, 0, width)
        y1 = np.clip(y_center - cut_h // 2, 0, height)
        x2 = np.clip(x_center + cut_w // 2, 0, width)
        y2 = np.clip(y_center + cut_h // 2, 0, height)
        
        # Create augmented batch
        augmented_images = []
        for i in range(batch_size):
            new_img = image_data[i].copy()
            # Paste the cut patch from the shuffled image
            new_img[y1:y2, x1:x2] = shuffled_images[i][y1:y2, x1:x2]
            augmented_images.append(new_img)
            
        return augmented_images