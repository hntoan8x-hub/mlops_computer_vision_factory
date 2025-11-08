# shared_libs/data_processing/image_components/feature_extractors/atomic/orb_extractor.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor, FeatureData, ImageData

logger = logging.getLogger(__name__)

class ORBExtractor(BaseFeatureExtractor):
    """
    Extracts ORB (Oriented FAST and Rotated BRIEF) features from an image batch.

    This component detects and computes keypoints and returns the descriptors.
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

    def extract(self, image: ImageData, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Finds and computes ORB descriptors from a single image or a list of images (batch).

        Args:
            image (ImageData): The input image(s).
            **kwargs: Additional keyword arguments.

        Returns:
            FeatureData: Descriptors as a NumPy array (single image) or a List[np.ndarray] (batch).
        
        Raises:
            TypeError: If input type is unsupported.
        """
        if isinstance(image, np.ndarray):
            # Handle single image
            return self._extract_single_image(image)
        elif isinstance(image, list):
            # Handle batch (list of images)
            return [self._extract_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _extract_single_image(self, img: np.ndarray) -> np.ndarray:
        """Helper to extract ORB features from a single image."""
        if len(img.shape) < 2:
            raise ValueError("Input image must have at least 2 dimensions.")
            
        # Convert to grayscale if the image is color
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = img

        try:
            # detectAndCompute returns keypoints (ignored) and descriptors
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            
            # Hardening: Return descriptors if found, otherwise an empty 2D array 
            # for consistent output shape/type when processing a batch.
            if descriptors is None:
                logger.debug("No ORB descriptors found for the image.")
                # ORB descriptors are typically float32; we return an empty array with 
                # the expected descriptor size if available, otherwise an empty array.
                descriptor_size = self.orb.descriptorSize() if hasattr(self.orb, 'descriptorSize') else 32 
                return np.array([], dtype=np.float32).reshape(0, descriptor_size)
            
            return descriptors
        except Exception as e:
            logger.error(f"Failed to extract ORB features. Error: {e}")
            raise