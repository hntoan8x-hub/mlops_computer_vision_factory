# shared_libs/data_processing/video_components/cleaners/video_frame_resizer.py

import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_video_cleaner import BaseVideoCleaner, VideoData

logger = logging.getLogger(__name__)

class VideoFrameResizer(BaseVideoCleaner):
    """
    Resizes all frames within a video sequence (4D tensor) to a specified width and height.

    This ensures dimensional consistency for downstream processing, including 
    frame sampling and image processing pipelines.
    """
    def __init__(self, width: int, height: int, interpolation: int = cv2.INTER_AREA):
        """
        Initializes the VideoFrameResizer.

        Args:
            width (int): The target width of the frames.
            height (int): The target height of the frames.
            interpolation (int, optional): The interpolation method for resizing (defaults to cv2.INTER_AREA).

        Raises:
            ValueError: If width or height are not positive integers.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        self.width = width
        self.height = height
        self.interpolation = interpolation
        logger.info(f"Initialized VideoFrameResizer to {self.width}x{self.height}.")

    def transform(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> VideoData:
        """
        Applies resizing to all frames in a single video or a list of videos (batch).

        Args:
            video (VideoData): The input video sequence(s) (T x H x W x C or List of 4D arrays).
            metadata (Optional[Dict[str, Any]]): Metadata (ignored by this atomic component).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            VideoData: The resized video sequence(s).
        
        Raises:
            TypeError: If input type is unsupported.
            ValueError: If the input array does not have 4 dimensions.
        """
        if isinstance(video, np.ndarray):
            return self._resize_single_video(video)
        elif isinstance(video, list):
            return [self._resize_single_video(vid) for vid in video]
        else:
            raise TypeError("Input must be a NumPy array (4D) or a list of NumPy arrays.")

    def _resize_single_video(self, video_arr: np.ndarray) -> np.ndarray:
        """
        Helper method to resize frames within a single video array.
        
        Args:
            video_arr (np.ndarray): The 4D video array (T x H x W x C).

        Returns:
            np.ndarray: The resized 4D video array.
        """
        if video_arr.ndim != 4:
            raise ValueError(f"Video input must be a 4D array (T x H x W x C). Found {video_arr.ndim} dimensions.")

        T, H, W, C = video_arr.shape
        resized_frames = []

        try:
            # Iterate through the time dimension (T) and resize each frame (H x W x C)
            for t in range(T):
                frame = video_arr[t]
                # Reuse cv2.resize logic from image cleaner, but applied per frame
                resized_frame = cv2.resize(frame, (self.width, self.height), interpolation=self.interpolation)
                resized_frames.append(resized_frame)
            
            # Stack frames back into a 4D array
            return np.stack(resized_frames)
            
        except Exception as e:
            logger.error(f"Failed to resize video frames. Expected output size: {self.height}x{self.width}. Error: {e}")
            raise