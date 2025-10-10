import abc
import numpy as np
from typing import Dict, Any, Union, List

# Type hint for image data, which can be a single image or a list of images.
ImageData = Union[np.ndarray, List[np.ndarray]]

class BaseImageCleaner(abc.ABC):
    """
    Abstract Base Class for image cleaning components.

    Defines a standard interface for preprocessing steps that clean raw image data.
    """

    @abc.abstractmethod
    def transform(self, image: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies a cleaning transformation to the input image(s).

        Args:
            image (ImageData): The input image or a list of images (e.g., from an IngestionOrchestrator).
            **kwargs: Additional parameters for the transformation.

        Returns:
            ImageData: The transformed image(s).
        """
        raise NotImplementedError