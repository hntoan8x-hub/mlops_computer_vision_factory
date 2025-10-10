import abc
import numpy as np
from typing import Dict, Any, Union, List

# Re-use the ImageData type hint for consistency.
ImageData = Union[np.ndarray, List[np.ndarray]]

class BaseAugmenter(abc.ABC):
    """
    Abstract Base Class for data augmentation components.

    Defines a standard interface for techniques that artificially increase
    the diversity of a dataset.
    """

    @abc.abstractmethod
    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies an augmentation transformation to the input image(s).

        Args:
            image (ImageData): The input image or a list of images.
            **kwargs: Additional parameters for the transformation.

        Returns:
            ImageData: The augmented image(s).
        """
        raise NotImplementedError