# shared_libs/data_processing/video_components/samplers/uniform_frame_sampler.py

import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_frame_sampler import BaseFrameSampler, VideoData, ImageData

logger = logging.getLogger(__name__)

class UniformFrameSampler(BaseFrameSampler):
    """
    Selects frames from a video sequence at a uniform interval or to reach a target count.

    This is the primary component for converting 4D VideoData into 3D ImageData.
    """
    def __init__(self, target_frames: Optional[int] = None, frame_interval: int = 1):
        """
        Initializes the UniformFrameSampler.

        Args:
            target_frames (Optional[int]): The exact number of frames to extract from the video. 
                                          If specified, it overrides frame_interval.
            frame_interval (int): The interval (step size) between sampled frames (e.g., 5 means sample every 5th frame).

        Raises:
            ValueError: If frame_interval is zero or negative.
        """
        if frame_interval <= 0:
            raise ValueError("Frame interval must be a positive integer.")
            
        self.target_frames = target_frames
        self.frame_interval = frame_interval
        
        logger.info(f"Initialized UniformFrameSampler. Target frames: {target_frames or 'None'}, Interval: {frame_interval}.")

    def sample(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Extracts frames from a single video or a list of videos (batch) using the configured policy.

        Args:
            video (VideoData): The input video sequence(s) (T x H x W x C or List of 4D arrays).
            metadata (Optional[Dict[str, Any]]): Metadata (ignored by this uniform component).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ImageData: A list of image frames (np.ndarray) extracted from the video.
        
        Raises:
            TypeError: If input type is unsupported.
        """
        if isinstance(video, np.ndarray):
            # Handle single video
            return self._sample_single_video(video)
        elif isinstance(video, list):
            # Handle batch of videos (will return a List[List[np.ndarray]])
            # NOTE: Orchestrator needs to flatten this structure if subsequent image processing is expected to run sequentially.
            return [self._sample_single_video(vid) for vid in video] 
        else:
            raise TypeError("Input must be a NumPy array (4D) or a list of NumPy arrays.")

    def _sample_single_video(self, video_arr: np.ndarray) -> List[np.ndarray]:
        """
        Helper method to sample frames from a single video array.
        
        Args:
            video_arr (np.ndarray): The 4D video array (T x H x W x C).

        Returns:
            List[np.ndarray]: The list of 3D image frames (ImageData).
        
        Raises:
            ValueError: If the input array does not have 4 dimensions.
        """
        if video_arr.ndim != 4:
            raise ValueError(f"Video input must be a 4D array (T x H x W x C). Found {video_arr.ndim} dimensions.")

        T = video_arr.shape[0]
        indices = []

        if self.target_frames is not None:
            # Policy 1: Target Frame Count (Overlaps/subsamples to match target)
            if T > 0:
                # Calculate indices to get exactly target_frames uniformly spread
                step = T / self.target_frames
                indices = [int(i * step) for i in range(self.target_frames)]
            
        else:
            # Policy 2: Fixed Interval (Subsampling)
            indices = list(range(0, T, self.frame_interval))
            
        # Hardening: Use np.take for efficient extraction of frames
        sampled_frames = np.take(video_arr, indices, axis=0)
        
        # Return as a standard Python list of 3D NumPy arrays (ImageData)
        return list(sampled_frames)