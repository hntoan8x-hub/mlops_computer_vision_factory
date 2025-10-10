import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor, FeatureData

logger = logging.getLogger(__name__)

class HOGExtractor(BaseFeatureExtractor):
    """
    Extracts HOG (Histogram of Oriented Gradients) features from an image.
    """
    def __init__(self, win_size: tuple = (64, 128), block_size: tuple = (16, 16),
                 block_stride: tuple = (8, 8), cell_size: tuple = (8, 8),
                 nbins: int = 9):
        """
        Initializes the HOGExtractor.

        Args:
            win_size (tuple): Size of the detection window.
            block_size (tuple): Size of the block.
            block_stride (tuple): Stride of the block.
            cell_size (tuple): Size of the cell.
            nbins (int): Number of bins for the orientation histogram.
        """
        self.hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )
        logger.info("Initialized HOGExtractor.")

    def extract(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Computes HOG features from a single image.

        Args:
            image (np.ndarray): The input image. Must be a single image.
            **kwargs: Additional keyword arguments.

        Returns:
            FeatureData: The HOG features as a NumPy array.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        try:
            # HOG requires a specific window size, which should be handled by an orchestrator or cleaner
            # For simplicity, we just check and resize if necessary.
            if gray_image.shape[1] != self.hog.winSize[0] or gray_image.shape[0] != self.hog.winSize[1]:
                gray_image = cv2.resize(gray_image, self.hog.winSize)
            
            # compute() returns a list of HOG features
            hog_features = self.hog.compute(gray_image)
            return hog_features.flatten()
        except Exception as e:
            logger.error(f"Failed to extract HOG features. Error: {e}")
            raise