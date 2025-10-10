import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor, FeatureData

logger = logging.getLogger(__name__)

class ORBExtractor(BaseFeatureExtractor):
    """
    Extracts ORB (Oriented FAST and Rotated BRIEF) features from an image.
    """
    def __init__(self, nfeatures: int = 500, scale_factor: float = 1.2, nlevels: int = 8):
        """
        Initializes the ORBExtractor.

        Args:
            nfeatures (int): The maximum number of features to retain.
            scale_factor (float): The pyramid decimation ratio.
            nlevels (int): The number of pyramid levels.
        """
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels
        )
        logger.info("Initialized ORBExtractor.")

    def extract(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Finds and computes ORB keypoints and descriptors from a single image.

        Args:
            image (np.ndarray): The input image. Must be a single image.
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
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            if descriptors is None:
                logger.warning("No ORB descriptors found for the image.")
            return descriptors
        except Exception as e:
            logger.error(f"Failed to extract ORB features. Error: {e}")
            raise