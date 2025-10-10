import numpy as np
import logging
from typing import Dict, Any, Union, List

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class Mixup(BaseAugmenter):
    """
    Implements the Mixup data augmentation technique.

    Mixes two images and their labels to create a new, blended image.
    This augmenter requires batches of data.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the Mixup augmenter.

        Args:
            alpha (float): The alpha parameter for the Beta distribution used to
                           determine the mixing ratio.
        """
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        self.alpha = alpha
        logger.info(f"Initialized Mixup with alpha={self.alpha}.")

    def transform(self, image_data: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies the Mixup augmentation to a batch of images.

        Args:
            image_data (ImageData): A list of images (a batch) to be augmented.
            **kwargs: Should contain 'labels' for the corresponding batch.

        Returns:
            ImageData: The augmented batch of images.
        """
        if not isinstance(image_data, list) or len(image_data) < 2:
            raise ValueError("Mixup requires a list of at least two images.")

        batch_size = len(image_data)
        
        # Check for labels in kwargs, which are required for Mixup
        if 'labels' not in kwargs:
            raise ValueError("Mixup augmentation requires a list of 'labels' in kwargs.")
        labels = kwargs['labels']

        # Choose a random image to mix with
        shuffle_indices = np.random.permutation(batch_size)
        shuffled_images = [image_data[i] for i in shuffle_indices]
        shuffled_labels = [labels[i] for i in shuffle_indices]

        # Generate lambda from a Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        augmented_images = []
        for i in range(batch_size):
            img1 = image_data[i]
            img2 = shuffled_images[i]
            
            # Mix the images
            new_img = lam * img1 + (1 - lam) * img2
            augmented_images.append(new_img.astype(img1.dtype))
            
        # The new labels are also a weighted mix.
        # This requires special handling in the loss function,
        # which is outside the scope of this augmenter.
        return augmented_images