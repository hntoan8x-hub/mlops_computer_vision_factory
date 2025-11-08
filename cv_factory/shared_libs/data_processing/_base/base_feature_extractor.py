# shared_libs/data_processing/_base/base_feature_extractor.py

import abc
import numpy as np
from typing import Dict, Any, Union, List

# Type hint for input image data (consistent with BaseImageCleaner and BaseEmbedder).
ImageData = Union[np.ndarray, List[np.ndarray]]

# Type hint for features, which can be a single array or a list of arrays.
FeatureData = Union[np.ndarray, List[np.ndarray]]

class BaseFeatureExtractor(abc.ABC):
    """
    Abstract Base Class for classical Computer Vision feature extractors (e.g., SIFT, HOG).

    Defines a standard interface for extracting handcrafted features from preprocessed images.
    Features are typically used as the primary input for traditional ML models or 
    as part of a feature engineering pipeline.
    """

    @abc.abstractmethod
    def extract(self, image: ImageData, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Extracts features from the input image(s).

        This method should handle both single images and batches (list of images).
        The features may represent keypoints, histograms, or other structured data.

        Args:
            image (ImageData): The input image(s), expected to be np.ndarray 
                               or List[np.ndarray] (batch).
            **kwargs: Additional parameters specific to the feature extraction algorithm 
                      (e.g., number of keypoints for SIFT).

        Returns:
            FeatureData: The extracted features. Typically a list of feature arrays 
                         if the output size varies per image (like SIFT keypoints), 
                         or a single aggregated numpy array if fixed-size (like HOG).
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError