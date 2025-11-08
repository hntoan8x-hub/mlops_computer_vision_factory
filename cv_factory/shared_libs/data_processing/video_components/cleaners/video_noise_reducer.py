# shared_libs/data_processing/video_components/cleaners/video_noise_reducer.py

import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional, Literal

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_video_cleaner import BaseVideoCleaner, VideoData

logger = logging.getLogger(__name__)

# Define supported filter types
FilterType = Literal["gaussian", "median", "bilateral"]

class VideoNoiseReducer(BaseVideoCleaner):
    """
    Applies noise reduction filters (e.g., Gaussian, Median, Bilateral) to every frame 
    within a video sequence to improve image quality for downstream processing.
    """
    
    def __init__(self, filter_type: FilterType = "gaussian", kernel_size: int = 5, sigma: float = 0.0):
        """
        Initializes the VideoNoiseReducer.

        Args:
            filter_type (FilterType): The type of filter to apply ("gaussian", "median", or "bilateral").
            kernel_size (int): The size of the kernel/window for the filter. Must be odd for Median/Gaussian.
            sigma (float): Standard deviation (sigma) for the Gaussian filter (ignored by others).

        Raises:
            ValueError: If kernel_size is invalid for the chosen filter type.
        """
        if filter_type in ["gaussian", "median"] and kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd for {filter_type} filter.")
            
        self.filter_type = filter_type
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        logger.info(f"Initialized VideoNoiseReducer with filter: {filter_type}, kernel: {kernel_size}.")

    def transform(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> VideoData:
        """
        Applies the noise reduction filter to all frames in a single video or a list of videos (batch).

        Args:
            video (VideoData): The input video sequence(s) (T x H x W x C or List of 4D arrays).
            metadata (Optional[Dict[str, Any]]): Metadata (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            VideoData: The noise-reduced video sequence(s).
        
        Raises:
            TypeError: If input type is unsupported.
        """
        if isinstance(video, np.ndarray):
            return self._reduce_noise_single_video(video)
        elif isinstance(video, list):
            return [self._reduce_noise_single_video(vid) for vid in video]
        else:
            raise TypeError("Input must be a NumPy array (4D) or a list of NumPy arrays.")

    def _reduce_noise_single_video(self, video_arr: np.ndarray) -> np.ndarray:
        """
        Helper method to apply noise reduction to frames within a single video array.
        """
        if video_arr.ndim != 4:
            raise ValueError("Video input must be a 4D array (T x H x W x C).")

        T = video_arr.shape[0]
        reduced_frames = []

        try:
            # Iterate through the time dimension (T) and apply filter per frame
            for t in range(T):
                frame = video_arr[t]
                
                if self.filter_type == "gaussian":
                    # Gaussian Blur: effective for general noise
                    reduced_frame = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), self.sigma)
                elif self.filter_type == "median":
                    # Median Blur: effective for salt-and-pepper noise
                    reduced_frame = cv2.medianBlur(frame, self.kernel_size)
                elif self.filter_type == "bilateral":
                    # Bilateral Filter: preserves edges while smoothing
                    # Need to convert frame to uint8 first, if it isn't already, for cv2 filter compatibility
                    temp_frame = frame.astype(np.uint8)
                    reduced_frame = cv2.bilateralFilter(temp_frame, self.kernel_size, self.sigma, self.sigma).astype(frame.dtype)
                else:
                    reduced_frame = frame # Should be caught by __init__ validation
                
                reduced_frames.append(reduced_frame)
            
            # Stack frames back into a 4D array
            return np.stack(reduced_frames)
            
        except Exception as e:
            logger.error(f"Failed to apply {self.filter_type} noise reduction filter. Error: {e}")
            raise