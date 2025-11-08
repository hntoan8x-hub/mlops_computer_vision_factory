# shared_libs/data_processing/video_components/samplers/key_frame_extractor.py

import logging
import numpy as np
import cv2
from typing import Dict, Any, Union, List, Optional, Literal

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_frame_sampler import BaseFrameSampler, VideoData, ImageData

logger = logging.getLogger(__name__)

# Supported metrics for measuring frame difference
MetricType = Literal["hist_diff", "ssim"]

class KeyFrameExtractor(BaseFrameSampler):
    """
    Selects key frames from a video sequence based on content difference (change detection).

    This sampler compares frames using a specified metric (e.g., histogram difference) 
    and extracts only those frames that exceed a certain threshold of change from the previous key frame.
    """
    
    def __init__(self, metric: MetricType = "hist_diff", threshold: float = 0.05, min_interval: int = 1):
        """
        Initializes the KeyFrameExtractor.

        Args:
            metric (MetricType): The metric used to measure difference ('hist_diff' or 'ssim').
            threshold (float): The minimum difference required to mark a frame as a key frame (0.0 to 1.0).
            min_interval (int): The minimum number of frames to skip before checking for a new key frame.

        Raises:
            ValueError: If the threshold or min_interval is invalid.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0.")
        if min_interval < 1:
            raise ValueError("Minimum interval must be 1 or greater.")
            
        self.metric = metric
        self.threshold = threshold
        self.min_interval = min_interval
        
        logger.info(f"Initialized KeyFrameExtractor. Metric: {metric}, Threshold: {threshold}.")

    def _calculate_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Helper to calculate the difference between two frames based on the configured metric."""
        
        if self.metric == "hist_diff":
            # Convert to grayscale and calculate normalized histograms
            if frame1.ndim == 3:
                frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                frame1_gray = frame1
                frame2_gray = frame2

            hist1 = cv2.calcHist([frame1_gray], [0], None, [256], [0, 256], accumulate=False)
            hist2 = cv2.calcHist([frame2_gray], [0], None, [256], [0, 256], accumulate=False)
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Use correlation (cv2.HISTCMP_CORREL) or Chi-Square (cv2.HISTCMP_CHISQR)
            # Correlation measures similarity (1 is identical). We need (1 - similarity) for difference.
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return 1.0 - similarity 
            
        elif self.metric == "ssim":
            # NOTE: SSIM requires scikit-image and is computationally expensive.
            # For simplicity, we use a basic difference calculation, but SSIM is preferred for accuracy.
            # Fallback to Mean Squared Error (MSE) for quick estimation if SSIM library is unavailable.
            mse = np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)
            # Normalize MSE to a 0-1 scale (assuming 8-bit image max difference is 255^2)
            max_mse = 255**2
            return mse / max_mse
        
        return 0.0


    def sample(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Extracts key frames from a single video (batch processing is delegated to list comprehension).

        Args:
            video (VideoData): The input video sequence(s) (expected to be a 4D NumPy array).
            metadata (Optional[Dict[str, Any]]): Metadata (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ImageData: The list of 3D key image frames extracted from the video.
        
        Raises:
            TypeError: If input type is unsupported.
        """
        if isinstance(video, np.ndarray):
            return self._sample_single_video(video)
        elif isinstance(video, list):
            # Handle batch by recursively calling the sampler (simplification for batch video processing)
            return [self._sample_single_video(vid) for vid in video] 
        else:
            raise TypeError("Input must be a NumPy array (4D) or a list of NumPy arrays.")

    def _sample_single_video(self, video_arr: np.ndarray) -> List[np.ndarray]:
        """
        Helper method to sample key frames from a single video array.
        """
        if video_arr.ndim != 4:
            raise ValueError("Video input must be a 4D array (T x H x W x C).")

        T = video_arr.shape[0]
        if T == 0:
            return []

        key_frames = [video_arr[0]] # Always include the first frame as the initial key frame
        last_key_frame = video_arr[0]
        
        # Start check after the minimum interval
        for t in range(self.min_interval, T):
            current_frame = video_arr[t]
            
            try:
                difference = self._calculate_difference(last_key_frame, current_frame)
                
                if difference >= self.threshold:
                    # Found a significant change, mark as a new key frame
                    key_frames.append(current_frame)
                    last_key_frame = current_frame
                    logger.debug(f"Frame {t} selected as key frame. Difference: {difference:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error calculating difference at frame {t}. Skipping frame. Error: {e}")
                # Continue processing even if one frame check fails
                continue

        return key_frames