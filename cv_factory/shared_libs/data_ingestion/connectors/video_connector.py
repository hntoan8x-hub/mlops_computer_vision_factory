# cv_factory/shared_libs/data_ingestion/connectors/video_connector.py

import logging
import cv2
import numpy as np
from typing import Dict, Any, Union, Optional
from ..base.base_data_connector import BaseDataConnector, OutputData

logger = logging.getLogger(__name__)

class VideoConnector(BaseDataConnector):
    """
    Concrete connector for reading and writing video files (static data).
    NOTE: For streaming video, use CameraStreamConnector.
    """
    
    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(connector_id, config)
        self.capture: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None

    def connect(self) -> bool:
        """Video connector does not require persistent external connection; initialization only."""
        self.is_connected = True
        return True

    def read(self, source_uri: str, as_frames: bool = True, **kwargs) -> OutputData:
        """
        Reads a video file and returns either a list of frames or the VideoCapture object.
        """
        self.capture = cv2.VideoCapture(source_uri)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Failed to open video file: {source_uri}")

        if not as_frames:
            return self.capture # Return the object for advanced handling
            
        frames = []
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break
            # Convert BGR to RGB before storing in the list
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        self.capture.release()
        return frames

    def write(self, data: OutputData, destination_uri: str, fps: float = 30.0, **kwargs) -> str:
        """
        Writes a list of frames (NumPy arrays) to a video file.
        
        Args:
            data (List[np.ndarray]): List of RGB frames to write.
            destination_uri (str): Path to the output video file (e.g., output.mp4).
            fps (float): Frames per second for the output video.
        """
        if not data or not isinstance(data[0], np.ndarray):
            raise ValueError("Data must be a non-empty list of NumPy arrays (frames).")
            
        height, width, _ = data[0].shape
        
        # Determine codec (e.g., 'mp4v' for .mp4)
        fourcc = cv2.VideoWriter_fourcc(*kwargs.get('codec', 'mp4v'))
        
        self.writer = cv2.VideoWriter(destination_uri, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise IOError(f"Could not initialize VideoWriter for path: {destination_uri}")

        for frame_rgb in data:
            # Convert frame back to BGR for OpenCV writing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)

        self.writer.release()
        logger.info(f"Video successfully written to {destination_uri}")
        return destination_uri

    def close(self):
        """Ensures any open capture or writer objects are released."""
        if self.capture: self.capture.release()
        if self.writer: self.writer.release()
        super().close()