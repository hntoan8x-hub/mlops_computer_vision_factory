# shared_libs/data_processing/video_components/_base/base_video_sampler.py

import abc
import numpy as np
from typing import Dict, Any, Union, List

# Re-use VideoData for input
VideoData = Union[np.ndarray, List[np.ndarray]]

# Re-use ImageData for output (consistent with BaseImageCleaner)
ImageData = Union[np.ndarray, List[np.ndarray]]

class BaseVideoSampler(abc.ABC):
    """
    Abstract Base Class for video frame sampling components.

    Defines a standard interface for converting continuous video sequences 
    (4D tensor) into discrete image frames (List of 3D tensors). 
    This is the bridge to the existing Image Processing pipeline.
    """

    @abc.abstractmethod
    def sample(self, video: VideoData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Selects and extracts representative frames from the input video(s).

        This process effectively converts the video data into image data for 
        downstream image cleaners, embedders, or models.

        Args:
            video (VideoData): The input video sequence(s).
            **kwargs: Parameters for the sampling strategy (e.g., sample rate, 
                      number of keyframes).

        Returns:
            ImageData: A single image or a list of image frames (np.ndarray) 
                       extracted from the video.
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError