# shared_libs/data_processing/video_components/_base/base_video_cleaner.py

import abc
import numpy as np
from typing import Dict, Any, Union, List, Optional

# Type hint for video data: a 4D array (T x H x W x C) or a list of 4D arrays (batch).
VideoData = Union[np.ndarray, List[np.ndarray]]

class BaseVideoCleaner(abc.ABC):
    """
    Abstract Base Class for video cleaning components.

    Defines a standard interface for preprocessing steps that clean 
    raw video data (e.g., resizing, frame rate adjustment) and operate 
    on the entire video sequence (4D tensor).
    """

    @abc.abstractmethod
    def transform(self, video: VideoData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> VideoData:
        """
        Applies a cleaning transformation to the input video(s).

        Args:
            video (VideoData): The input video sequence(s).
            metadata (Optional[Dict[str, Any]]): Metadata (e.g., source FPS) used for adaptive processing.
            **kwargs: Additional parameters for the transformation.

        Returns:
            VideoData: The transformed video sequence(s).
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError