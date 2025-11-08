# shared_libs/data_processing/video_components/cleaners/motion_stabilizer.py

import cv2
import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional, Literal

# Import Abstractions
from shared_libs.data_processing.video_components._base.base_video_cleaner import BaseVideoCleaner, VideoData

logger = logging.getLogger(__name__)

class MotionStabilizer(BaseVideoCleaner):
    """
    Applies video stabilization techniques to reduce camera shake/motion between consecutive frames.

    This process estimates the transformation (e.g., translation, rotation, scale) 
    between frames and compensates for it.
    """
    
    def __init__(self, border_mode: Literal["replicate", "constant"] = "replicate"):
        """
        Initializes the MotionStabilizer.

        Args:
            border_mode (Literal["replicate", "constant"]): Defines how to fill the borders 
                that appear after stabilization (e.g., "replicate" copies the edge pixels).
        """
        self.border_mode = border_mode
        self.border_code = self._get_border_code(border_mode)
        
        # NOTE: Using a simplified stabilization approach (translation) for atomic component logic.
        self.max_corners = 2000 # Max number of features to track
        self.feature_params = dict(maxCorners=self.max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        logger.info(f"Initialized MotionStabilizer with border mode: {border_mode}.")

    def _get_border_code(self, mode: str) -> int:
        """Maps border string to OpenCV constant."""
        if mode == "replicate":
            return cv2.BORDER_REPLICATE
        elif mode == "constant":
            return cv2.BORDER_CONSTANT
        raise ValueError(f"Unsupported border mode: {mode}")

    def transform(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> VideoData:
        """
        Applies motion stabilization to a single video or a list of videos (batch).

        Args:
            video (VideoData): The input video sequence(s).
            metadata (Optional[Dict[str, Any]]): Metadata (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            VideoData: The motion-stabilized video sequence(s).
        """
        if isinstance(video, np.ndarray):
            return self._stabilize_single_video(video)
        elif isinstance(video, list):
            return [self._stabilize_single_video(vid) for vid in video]
        else:
            raise TypeError("Input must be a NumPy array (4D) or a list of NumPy arrays.")

    def _stabilize_single_video(self, video_arr: np.ndarray) -> np.ndarray:
        """
        Helper method to stabilize frames within a single video array using iterative transformation.
        """
        if video_arr.ndim != 4:
            raise ValueError("Video input must be a 4D array (T x H x W x C).")

        T, H, W, C = video_arr.shape
        if T < 2:
            logger.warning("Video has less than 2 frames. Skipping stabilization.")
            return video_arr

        stabilized_frames = []
        
        # Convert first frame to grayscale (for feature tracking)
        prev_frame = video_arr[0]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY) if C == 3 else prev_frame
        
        # The initial frame is unchanged
        stabilized_frames.append(prev_frame) 
        
        # Accumulate the total transformation
        accumulated_transform = np.zeros((2, 3), dtype=np.float32)

        for t in range(1, T):
            curr_frame = video_arr[t]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY) if C == 3 else curr_frame
            
            # 1. Feature Detection (on previous frame)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
            
            if p0 is None or len(p0) < 3:
                logger.debug(f"Frame {t}: Not enough features for stabilization. Using identity transform.")
                # If stabilization fails, use the last valid frame/identity transform
                stabilized_frames.append(curr_frame)
                prev_frame = curr_frame
                prev_gray = curr_gray
                continue

            # 2. Feature Tracking (Optical Flow)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
            
            # Filter good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) < 3:
                # Fallback if tracking failed
                stabilized_frames.append(curr_frame)
                prev_frame = curr_frame
                prev_gray = curr_gray
                continue

            # 3. Estimate Transformation (Translation/Rotation/Scale)
            # Use estimateAffine2D for rigid body motion (translation + rotation + scale)
            # This returns a 2x3 transformation matrix
            transform_matrix, inliers = cv2.estimateAffine2D(good_old, good_new)
            
            if transform_matrix is None:
                # Fallback if matrix estimation fails
                stabilized_frames.append(curr_frame)
                prev_frame = curr_frame
                prev_gray = curr_gray
                continue

            # 4. Correct (Invert) Motion
            # We only use the translation component for simplicity in an atomic component
            dx = transform_matrix[0, 2]
            dy = transform_matrix[1, 2]
            
            # The corrected transformation matrix (2x3) to apply to *all* subsequent frames
            # This ensures all frames are registered to the first frame.
            accumulated_transform[0, 2] += dx
            accumulated_transform[1, 2] += dy
            
            # Create the inverse accumulated warp matrix
            # We want to warp the current frame back to the registration of the first frame
            M = np.array([[1, 0, accumulated_transform[0, 2]], [0, 1, accumulated_transform[1, 2]]], dtype=np.float32)

            # 5. Apply Warp
            stabilized_frame = cv2.warpAffine(curr_frame, M, (W, H), borderMode=self.border_code)
            stabilized_frames.append(stabilized_frame)

            # Update for next iteration
            prev_frame = curr_frame
            prev_gray = curr_gray

        # Stack frames back into a 4D array
        return np.stack(stabilized_frames)