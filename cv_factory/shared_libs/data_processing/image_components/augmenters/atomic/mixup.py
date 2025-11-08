# shared_libs/data_processing/image_components/augmenters/atomic/mixup.py

import numpy as np
import logging
from typing import Dict, Any, Union, List

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class Mixup(BaseAugmenter):
    """
    Implements the Mixup data augmentation technique.

    Requires batches of data and their labels for linear blending.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the Mixup augmenter.
        """
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        self.alpha = alpha
        logger.info(f"Initialized Mixup with alpha={self.alpha}.")

    def transform(self, image_data: ImageData, magnitude: float = 1.0, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies the Mixup augmentation to a batch of images.

        Args:
            image_data (ImageData): A list of images (a batch) to be augmented.
            magnitude (float): The intensity of the transformation (0.0 to 1.0). 
                               Used for interface compliance (can be used to bias alpha/lambda).
            **kwargs: Must contain 'labels' (List or np.ndarray) for the corresponding batch.

        Returns:
            ImageData: The augmented batch of images.
            
        Raises:
            ValueError: If the input batch size is insufficient or labels are missing/inconsistent.
        """
        if not isinstance(image_data, list) or len(image_data) < 2:
            raise ValueError("Mixup requires a list of at least two images (a batch).")

        batch_size = len(image_data)
        
        # HARDENING: Label Check
        if 'labels' not in kwargs:
            raise ValueError("Mixup augmentation requires a list or array of 'labels' in kwargs.")
        labels = kwargs['labels']
        if len(labels) != batch_size:
             raise ValueError(f"Label count ({len(labels)}) does not match batch size ({batch_size}).")

        # Convert images to float for safe arithmetic mixing
        float_images = [img.astype(np.float32) for img in image_data]
        shuffle_indices = np.random.permutation(batch_size)
        shuffled_images = [float_images[i] for i in shuffle_indices]
        
        # Generate lambda from a Beta distribution (lambda determines the mixing ratio)
        lam = np.random.beta(self.alpha, self.alpha)

        augmented_images = []
        for i in range(batch_size):
            img1 = float_images[i]
            img2 = shuffled_images[i]
            
            # Mix the images
            new_img = lam * img1 + (1 - lam) * img2
            
            # Convert back to original type, clipping if necessary
            if image_data[i].dtype == np.uint8:
                augmented_images.append(np.clip(new_img, 0, 255).astype(np.uint8))
            else:
                augmented_images.append(new_img.astype(image_data[i].dtype))
            
        return augmented_images