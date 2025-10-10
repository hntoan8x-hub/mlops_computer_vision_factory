import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor, FeatureData

logger = logging.getLogger(__name__)

class SIFTExtractor(BaseFeatureExtractor):
    """
    Extracts SIFT (Scale-Invariant Feature Transform) features from an image.
    """
    def __init__(self, nfeatures: int = 0, n_octave_layers: int = 3, contrast_threshold: float = 0.04):
        """
        Initializes the SIFTExtractor.

        Args:
            nfeatures (int): The number of best features to retain.
            n_octave_layers (int): The number of layers in each octave.
            contrast_threshold (float): Threshold to filter out weak features.
        """
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold
        )
        logger.info("Initialized SIFTExtractor.")

    def extract(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Finds and computes SIFT keypoints and descriptors from a single image.

        Args:
            image (np.ndarray): The input image. Must be a single image (not a list).
            **kwargs: Additional keyword arguments.

        Returns:
            FeatureData: A tuple containing keypoints (List[cv2.KeyPoint]) and descriptors (np.ndarray).
                         Returns None for descriptors if no keypoints are found.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        try:
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
            if descriptors is None:
                logger.warning("No SIFT descriptors found for the image.")
            return descriptors
        except Exception as e:
            logger.error(f"Failed to extract SIFT features. Error: {e}")
            raise