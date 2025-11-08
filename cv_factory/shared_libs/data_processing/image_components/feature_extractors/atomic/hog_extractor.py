# shared_libs/data_processing/image_components/feature_extractors/atomic/hog_extractor.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor, FeatureData, ImageData

logger = logging.getLogger(__name__)

class HOGExtractor(BaseFeatureExtractor):
    """
    Extracts HOG (Histogram of Oriented Gradients) features from an image batch.

    The component includes internal resizing logic to match the required win_size.
    """
    def __init__(self, win_size: tuple = (64, 128), block_size: tuple = (16, 16),
                 block_stride: tuple = (8, 8), cell_size: tuple = (8, 8),
                 nbins: int = 9):
        """
        Initializes the HOGExtractor.

        Args:
            win_size (tuple): Size of the detection window (Width, Height).
            block_size (tuple): Size of the block.
            block_stride (tuple): Stride of the block.
            cell_size (tuple): Size of the cell.
            nbins (int): Number of bins for the orientation histogram.
        """
        # OpenCV HOGDescriptor expects (Width, Height)
        self.hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )
        logger.info(f"Initialized HOGExtractor with win_size: {win_size}.")

    def extract(self, image: ImageData, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Computes HOG features from a single image or a list of images (batch).

        Args:
            image (ImageData): The input image(s).
            **kwargs: Additional keyword arguments.

        Returns:
            FeatureData: The HOG features as a NumPy array (single image) or a List[np.ndarray] (batch).
        
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
        """Helper to compute HOG features from a single image."""
        if len(img.shape) < 2:
            raise ValueError("Input image must have at least 2 dimensions.")
            
        # Convert to grayscale if the image is color
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = img

        # Hardening: Check and resize the image to match HOG window size (if necessary)
        win_w, win_h = self.hog.winSize
        if gray_image.shape[1] != win_w or gray_image.shape[0] != win_h:
            logger.debug(f"Resizing image from {gray_image.shape[1]}x{gray_image.shape[0]} to {win_w}x{win_h} for HOG.")
            gray_image = cv2.resize(gray_image, (win_w, win_h))
        
        try:
            # compute() returns a list of HOG features
            hog_features = self.hog.compute(gray_image)
            
            # Flatten to a 1D vector (common practice for HOG feature vectors)
            return hog_features.flatten()
        except Exception as e:
            logger.error(f"Failed to extract HOG features. Error: {e}")
            # Hardening: Return an empty array on failure for batch consistency
            raise