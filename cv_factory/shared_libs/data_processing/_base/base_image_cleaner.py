# shared_libs/data_processing/_base/base_image_cleaner.py
import abc
import numpy as np
from typing import Dict, Any, Union, List

# Type hint for image data, which can be a single image or a list of images.
ImageData = Union[np.ndarray, List[np.ndarray]]

class BaseImageCleaner(abc.ABC):
    """
    Abstract Base Class for image cleaning components (Preprocessing).

    Defines a standard interface for essential preprocessing steps that clean 
    and standardize raw image data (e.g., resizing, normalization, color space conversion).
    """

    @abc.abstractmethod
    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies a cleaning transformation to the input image(s).

        This method should handle both single images and batches (list of images).

        Args:
            image (ImageData): The input image or a list of images (e.g., raw data 
                               from a Data Connector).
            **kwargs: Additional parameters for the transformation (e.g., interpolation 
                      method for resizing).

        Returns:
            ImageData: The transformed image(s).
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError