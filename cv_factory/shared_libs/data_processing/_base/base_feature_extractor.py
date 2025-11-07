# shared_libs/data_processing/_base/base_feature_extractor.py

import abc
import numpy as np
from typing import Dict, Any, Union, List

# Type hint for features, which can be a single array or a list of arrays.
FeatureData = Union[np.ndarray, List[np.ndarray]]

class BaseFeatureExtractor(abc.ABC):
    """
    Abstract Base Class for classical Computer Vision feature extractors.

    Defines a standard interface for extracting handcrafted features like SIFT or HOG.
    """

    @abc.abstractmethod
    def extract(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Extracts features from a single image.

        Args:
            image (np.ndarray): The input image.
            **kwargs: Additional parameters for feature extraction.

        Returns:
            FeatureData: The extracted features.
        """
        raise NotImplementedError