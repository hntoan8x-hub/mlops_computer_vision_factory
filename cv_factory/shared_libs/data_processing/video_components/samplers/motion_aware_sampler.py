# shared_libs/data_processing/video_components/samplers/motion_aware_sampler.py

import logging
import numpy as np
import cv2
from typing import Dict, Any, Union, List, Optional

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_frame_sampler import BaseFrameSampler, VideoData, ImageData

logger = logging.getLogger(__name__)

class MotionAwareSampler(BaseFrameSampler):
    """
    Selects frames from a video sequence based on the magnitude of inter-frame motion.

    This sampler uses a simplified motion detection method (e.g., frame difference or 
    Optical Flow analysis) to prioritize frames with significant activity.
    """
    
    def __init__(self, motion_threshold: float = 5.0, min_motion_frames: int = 10):
        """
        Initializes the MotionAwareSampler.

        Args:
            motion_threshold (float): The minimum average difference/flow magnitude required 
                                      to mark a frame as containing significant motion.
            min_motion_frames (int): The minimum number of frames required to see motion 
                                     before selecting the next motion frame.

        Raises:
            ValueError: If the threshold is negative or min_motion_frames is invalid.
        """
        if motion_threshold <= 0:
            raise ValueError("Motion threshold must be positive.")
        if min_motion_frames < 1:
            raise ValueError("Minimum motion frames interval must be 1 or greater.")
            
        self.motion_threshold = motion_threshold
        self.min_motion_frames = min_motion_frames
        
        # Parameters for simplified Optical Flow calculation (Farneback or simple frame diff)
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        logger.info(f"Initialized MotionAwareSampler. Threshold: {motion_threshold}.")

    def _calculate_motion_magnitude(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Helper to calculate the magnitude of motion (e.g., average optical flow magnitude) 
        between two consecutive frames.
        """
        if frame1.ndim == 3:
            prev_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = frame1
            curr_gray = frame2

        # Ensure types are appropriate for cv2.calcOpticalFlowFarneback
        prev_gray = prev_gray.astype(np.uint8)
        curr_gray = curr_gray.astype(np.uint8)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **self.flow_params)
        
        # Calculate the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Return the average magnitude of motion
        return np.mean(magnitude)


    def sample(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Extracts frames that contain significant motion from a single video (batch processing delegated).

        Args:
            video (VideoData): The input video sequence(s) (expected to be a 4D NumPy array).
            metadata (Optional[Dict[str, Any]]): Metadata (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ImageData: The list of 3D image frames extracted from the video.
        
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
        Helper method to sample motion-aware frames from a single video array.
        """
        if video_arr.ndim != 4:
            raise ValueError("Video input must be a 4D array (T x H x W x C).")

        T = video_arr.shape[0]
        if T == 0:
            return []

        sampled_frames = [video_arr[0]] # Always include the first frame
        last_motion_frame_index = 0
        
        # Start checking from the second frame
        for t in range(1, T):
            
            # Check the minimum interval since the last sampled motion frame
            if t - last_motion_frame_index < self.min_motion_frames:
                continue
                
            prev_frame = video_arr[t - 1]
            current_frame = video_arr[t]
            
            try:
                # Calculate motion between the current frame and the immediate previous frame
                motion = self._calculate_motion_magnitude(prev_frame, current_frame)
                
                if motion >= self.motion_threshold:
                    # Found significant motion, mark as a sampled frame
                    sampled_frames.append(current_frame)
                    last_motion_frame_index = t
                    logger.debug(f"Frame {t} selected due to motion. Magnitude: {motion:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error calculating motion at frame {t}. Skipping frame. Error: {e}")
                continue

        return sampled_frames