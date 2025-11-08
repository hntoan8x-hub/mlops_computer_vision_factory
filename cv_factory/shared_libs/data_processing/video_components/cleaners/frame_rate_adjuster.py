# shared_libs/data_processing/video_components/cleaners/frame_rate_adjuster.py

import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_video_cleaner import BaseVideoCleaner, VideoData

logger = logging.getLogger(__name__)

class FrameRateAdjuster(BaseVideoCleaner):
    """
    Adjusts the effective frame rate of a video sequence by sub-sampling frames.

    This component is typically used to reduce computational load or to standardize 
    the temporal dimension of videos before sampling.
    """
    
    def __init__(self, target_fps: Optional[float] = None, interval: Optional[int] = None):
        """
        Initializes the FrameRateAdjuster.

        Args:
            target_fps (Optional[float]): The target frame rate (FPS) to simulate. Requires metadata 
                                          containing the original FPS for calculation.
            interval (Optional[int]): The fixed step interval (e.g., 3 means keep every 3rd frame). 
                                      If target_fps is provided, interval is ignored.

        Raises:
            ValueError: If neither target_fps nor interval is provided, or if interval is invalid.
        """
        if target_fps is None and (interval is None or interval <= 0):
            raise ValueError("Must provide a positive 'interval' or a 'target_fps'.")
            
        self.target_fps = target_fps
        self.interval = interval
        
        logger.info(f"Initialized FrameRateAdjuster. Mode: {self.target_fps or self.interval}.")

    def transform(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> VideoData:
        """
        Applies frame rate adjustment to a single video or a list of videos (batch).

        Args:
            video (VideoData): The input video sequence(s) (T x H x W x C or List of 4D arrays).
            metadata (Optional[Dict[str, Any]]): Metadata, expected to contain 'source_fps' if target_fps is set.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            VideoData: The adjusted video sequence(s).
        
        Raises:
            RuntimeError: If necessary metadata (source_fps) is missing.
        """
        if isinstance(video, np.ndarray):
            return self._adjust_single_video(video, metadata)
        elif isinstance(video, list):
            return [self._adjust_single_video(vid, metadata) for vid in video]
        else:
            raise TypeError("Input must be a NumPy array (4D) or a list of NumPy arrays.")

    def _adjust_single_video(self, video_arr: np.ndarray, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Helper method to adjust frames within a single video array.
        """
        if video_arr.ndim != 4:
            raise ValueError("Video input must be a 4D array (T x H x W x C).")

        T = video_arr.shape[0]
        step_interval = self.interval

        if self.target_fps is not None:
            source_fps = metadata.get('source_fps') if metadata else None
            if not source_fps or source_fps <= 0:
                raise RuntimeError("Target FPS mode requires valid 'source_fps' in metadata.")
                
            # Calculate interval based on FPS ratio (Hardening)
            step_interval = max(1, int(source_fps / self.target_fps))
            logger.debug(f"Adjusting FPS from {source_fps} to {self.target_fps}. Calculated interval: {step_interval}")

        if step_interval == 1:
            return video_arr # No change needed

        # Select indices based on the calculated step interval
        indices = list(range(0, T, step_interval))
        
        # Hardening: Use np.take for efficient extraction of frames along the time axis
        adjusted_video = np.take(video_arr, indices, axis=0)
        
        return adjusted_video