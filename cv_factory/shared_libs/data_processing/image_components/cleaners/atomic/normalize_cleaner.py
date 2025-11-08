# shared_libs/data_processing/image_components/cleaners/atomic/normalize_cleaner.py
import numpy as np
import logging
from typing import Dict, Any, Union, List, Optional

from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner, ImageData

logger = logging.getLogger(__name__)

class NormalizationError(Exception):
    """Custom exception for errors during normalization process."""
    pass

class NormalizeCleaner(BaseImageCleaner):
    """
    Normalizes image pixel values using the formula: normalized_pixel = (pixel - mean) / std.

    Ensures mean and std match image channel dimensions.
    """
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        """
        Initializes the NormalizeCleaner.

        Args:
            mean (Union[float, List[float]]): The mean value(s) for normalization.
            std (Union[float, List[float]]): The standard deviation value(s) for normalization.

        Raises:
            ValueError: If the shape of mean and std are not identical.
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        if self.mean.shape != self.std.shape:
             raise ValueError("The shape of 'mean' and 'std' must be identical.")
             
        logger.info(f"Initialized NormalizeCleaner with mean={self.mean} and std={self.std}.")

    def transform(self, image: ImageData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Normalizes a single image or a list of images.

        Args:
            image (ImageData): The input image(s).
            metadata (Optional[Dict[str, Any]]): Metadata of the input image (unused but kept for interface consistency).
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The normalized image(s).
        """
        if isinstance(image, np.ndarray):
            return self._normalize_single_image(image)
        elif isinstance(image, list):
            return [self._normalize_single_image(img) for img in image]
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")

    def _normalize_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Helper method to normalize a single image, including Input Shape Validation.
        """
        try:
            # 1. HARDENING: Input Shape Validation
            if img.ndim not in [2, 3]:
                raise NormalizationError(f"Unsupported image dimensions: {img.ndim}. Must be 2D (HxW) or 3D (HxWxC).")

            # 2. HARDENING: Consistency Check (Mean/Std vs. Channels)
            img_channels = 1 if img.ndim == 2 else img.shape[-1]
            mean_length = self.mean.size

            if img.ndim == 3 and img_channels != mean_length:
                 raise NormalizationError(
                    f"Channel mismatch: Image has {img_channels} channel(s), "
                    f"but {self.__class__.__name__} was initialized with {mean_length} mean/std values."
                 )
            elif img.ndim == 2 and mean_length not in [1, img_channels]: # img_channels = 1
                 raise NormalizationError(
                    f"Grayscale image (2D) requires 1 mean/std value, but {mean_length} values were provided."
                 )
                 
            # Convert image to float32 before normalization
            img = img.astype(np.float32)
            
            normalized_img = (img - self.mean) / self.std
            
            # 3. HARDENING: Range Check (optional, for debugging/logging)
            min_val, max_val = np.min(normalized_img), np.max(normalized_img)
            if abs(min_val) > 10.0 or abs(max_val) > 10.0:
                 logger.warning(
                    f"Normalized image values seem extreme (min: {min_val:.2f}, max: {max_val:.2f}). "
                    "Check input data for potential outliers."
                 )
            
            return normalized_img
        except NormalizationError as ne:
            raise ne
        except Exception as e:
            logger.error(f"Failed to normalize image. Original Error: {e}")
            raise NormalizationError(f"Internal error during normalization: {e}")