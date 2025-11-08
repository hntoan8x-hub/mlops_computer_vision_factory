# shared_libs/data_processing/_base/base_augmenter.py
import abc
import numpy as np
from typing import Dict, Any, Union, List

# Re-use the ImageData type hint for consistency.
ImageData = Union[np.ndarray, List[np.ndarray]]

class BaseAugmenter(abc.ABC):
    """
    Abstract Base Class for data augmentation components.

    Defines a standard interface for techniques that artificially increase
    the diversity of a dataset (e.g., rotation, noise injection, CutMix).
    This class is typically only used during the training lifecycle.
    """

    @abc.abstractmethod
    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies an augmentation transformation to the input image(s).

        This method should handle both single images and batches (list of images).
        The signature is intentionally kept identical to BaseImageCleaner for pipeline 
        consistency.

        Args:
            image (ImageData): The input image or a list of images.
            **kwargs: Additional parameters for the transformation.

        Returns:
            ImageData: The augmented image(s).
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError