# cv_factory/shared_libs/data_ingestion/connectors/video_connector.py

import logging
import cv2
import numpy as np
from typing import Dict, Any, Union, Optional, List
from ..base.base_data_connector import BaseDataConnector, OutputData

logger = logging.getLogger(__name__)

class VideoConnector(BaseDataConnector):
    """
    Concrete connector for reading and writing video files (static data).
    
    This connector uses OpenCV's VideoCapture for reading and VideoWriter for 
    writing video files, ensuring frames are consistently converted to RGB format.
    
    NOTE: For streaming video, the CameraStreamConnector should be used.
    """
    
    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the VideoConnector.

        Args:
            connector_id: A unique identifier for this connector instance.
            config: Configuration settings for the connection.
        """
        super().__init__(connector_id, config)
        self.capture: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None

    def connect(self) -> bool:
        """
        Video connector does not require a persistent external connection; 
        initialization only.

        Returns:
            bool: Always True.
        """
        self.is_connected = True
        logger.info(f"[{self.connector_id}] Video Connector initialized.")
        return True

    def read(self, source_uri: str, as_frames: bool = True, **kwargs) -> OutputData:
        """
        Reads a video file and returns either a list of frames or the VideoCapture object.

        Args:
            source_uri: The URI or path to the video file.
            as_frames: If True, returns a list of RGB NumPy arrays. 
                       If False, returns the raw cv2.VideoCapture object.
            **kwargs: Optional custom parameters.

        Returns:
            OutputData: A list of image arrays (RGB) or the VideoCapture object.

        Raises:
            FileNotFoundError: If the video file cannot be opened.
        """
        self.capture = cv2.VideoCapture(source_uri)
        
        # Hardening: Check if the file was truly opened
        if not self.capture.isOpened():
            logger.error(f"[{self.connector_id}] Failed to open video file: {source_uri}")
            raise FileNotFoundError(f"Failed to open video file: {source_uri}")

        if not as_frames:
            return self.capture 
            
        frames: List[np.ndarray] = []
        frame_count = 0
        
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            
            if not ret:
                break # End of file or read error
            
            # Convert BGR (OpenCV default) to RGB (ML standard)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
            
        self.capture.release()
        logger.info(f"[{self.connector_id}] Successfully read {frame_count} frames from {source_uri}.")
        
        if not frames:
            logger.warning(f"[{self.connector_id}] Read was successful, but no frames were extracted from {source_uri}.")
            
        return frames

    def write(self, data: OutputData, destination_uri: str, fps: float = 30.0, **kwargs) -> str:
        """
        Writes a list of frames (NumPy arrays) to a video file.
        
        Args:
            data: List of RGB frames (np.ndarray) to write.
            destination_uri: Path to the output video file (e.g., output.mp4).
            fps: Frames per second for the output video.
            **kwargs: Optional parameters, including 'codec' (e.g., 'mp4v', 'XVID').
            
        Returns:
            str: The final path/URI of the written video file.

        Raises:
            ValueError: If data is not a valid list of frames.
            IOError: If VideoWriter cannot be initialized.
        """
        # Hardening: Input validation
        if not isinstance(data, list) or not data or not isinstance(data[0], np.ndarray):
            raise ValueError("Data must be a non-empty list of NumPy arrays (frames) for writing.")
            
        first_frame = data[0]
        if first_frame.ndim != 3:
             raise ValueError("Frames must be 3-dimensional (H, W, C).")
             
        height, width, channels = first_frame.shape
        
        if channels != 3:
            logger.warning(f"Writing non-RGB video with {channels} channels.")
        
        # Determine codec (e.g., 'mp4v' for .mp4, 'XVID' for .avi)
        codec = kwargs.get('codec', 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        self.writer = cv2.VideoWriter(destination_uri, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            logger.error(f"[{self.connector_id}] Could not initialize VideoWriter with codec '{codec}' for path: {destination_uri}")
            raise IOError(f"Could not initialize VideoWriter for path: {destination_uri}")

        frame_written_count = 0
        for frame_rgb in data:
            # Convert frame back to BGR for OpenCV writing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)
            frame_written_count += 1

        self.writer.release()
        logger.info(f"[{self.connector_id}] Video successfully written to {destination_uri} with {frame_written_count} frames.")
        return destination_uri

    def close(self):
        """
        Ensures any open capture or writer objects are released safely.
        """
        if self.capture and self.capture.isOpened(): 
            self.capture.release()
            logger.debug(f"[{self.connector_id}] VideoCapture released.")
            
        if self.writer and self.writer.isOpened(): 
            self.writer.release()
            logger.debug(f"[{self.connector_id}] VideoWriter released.")
            
        super().close()