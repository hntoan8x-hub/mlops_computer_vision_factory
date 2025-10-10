import numpy as np
import logging
from typing import Dict, Any, Union, List, Tuple

from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData

logger = logging.getLogger(__name__)

class CutMix(BaseAugmenter):
    """
    Implements the CutMix data augmentation technique.

    This augmenter requires batches of data to mix.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the CutMix augmenter.

        Args:
            alpha (float): The alpha parameter for the Beta distribution used to
                           determine the mixing ratio. A higher alpha (e.g., > 1)
                           leads to a more equal split, while a lower alpha
                           (e.g., < 1) leads to more extreme splits.
        """
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        self.alpha = alpha
        logger.info(f"Initialized CutMix with alpha={self.alpha}.")

    def transform(self, image_data: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies the CutMix augmentation to a batch of images.

        Args:
            image_data (ImageData): A list of images (a batch) to be augmented.
            **kwargs: Should contain 'labels' for the corresponding batch.

        Returns:
            ImageData: The augmented batch of images.
        """
        if not isinstance(image_data, list) or len(image_data) < 2:
            raise ValueError("CutMix requires a list of at least two images.")

        batch_size = len(image_data)
        
        # Check for labels in kwargs, which are required for CutMix
        if 'labels' not in kwargs:
            raise ValueError("CutMix augmentation requires a list of 'labels' in kwargs.")
        labels = kwargs['labels']

        # Choose a random image to mix with
        shuffle_indices = np.random.permutation(batch_size)
        shuffled_images = [image_data[i] for i in shuffle_indices]
        shuffled_labels = [labels[i] for i in shuffle_indices]

        # Generate lambda from a Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Calculate the bounding box to be cut and pasted
        width, height, _ = image_data[0].shape
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Get random coordinates for the bounding box
        x1 = np.random.randint(width)
        y1 = np.random.randint(height)
        x2 = np.clip(x1 + cut_w, 0, width)
        y2 = np.clip(y1 + cut_h, 0, height)
        
        # Create augmented batch
        augmented_images = []
        for i in range(batch_size):
            new_img = image_data[i].copy()
            new_img[y1:y2, x1:x2] = shuffled_images[i][y1:y2, x1:x2]
            augmented_images.append(new_img)
            
        # The new labels would be a mix of the original and shuffled labels,
        # weighted by lam. This requires special handling in the loss function,
        # which is outside the scope of this augmenter. We just return the new images.
        return augmented_images