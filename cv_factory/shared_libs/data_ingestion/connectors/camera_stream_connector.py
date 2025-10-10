# cv_factory/shared_libs/data_ingestion/connectors/camera_stream_connector.py

import logging
import cv2
import time
from typing import Dict, Any, Iterator, Optional, Union, Tuple

# Import Base Abstraction
from ..base.base_stream_connector import BaseStreamConnector, StreamData 
# BaseStreamConnector enforces connect, consume, produce, close

logger = logging.getLogger(__name__)

class CameraStreamConnector(BaseStreamConnector):
    """
    A concrete stream connector for real-time camera or video file streaming. 
    
    Acts as a Consumer (reads frames) and a Producer (writes output video/results), 
    adhering strictly to the BaseStreamConnector contract for safe resource management.
    """

    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CameraStreamConnector.

        Args:
            connector_id (str): Unique ID.
            config (Optional[Dict[str, Any]]): Configuration including 'source' (int/str) 
                                               and 'output_path'/'output_config'.
        """
        super().__init__(connector_id, config)
        self.source = self.config.get('source')  # Camera index (0, 1, ...) or video file path
        self.output_path = self.config.get('output_path')
        self.output_config = self.config.get('output_config', {})
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_metadata: Dict[str, Any] = {} # Store properties like FPS, frame size
        
    def connect(self) -> bool:
        """
        Initializes the OpenCV VideoCapture object for the input stream.
        """
        if self.is_connected:
            return True
            
        if self.source is None:
            raise ValueError("Camera source must be specified in config.")

        try:
            # Initialize Video Capture
            self.video_capture = cv2.VideoCapture(self.source)
            
            if not self.video_capture.isOpened():
                raise IOError("Could not open video stream or camera source.")
                
            # Store frame properties for downstream use (e.g., initializing VideoWriter)
            self.frame_metadata['fps'] = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_metadata['width'] = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_metadata['height'] = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.is_connected = True
            logger.info(f"[{self.connector_id}] Camera stream connected. Resolution: {self.frame_metadata['width']}x{self.frame_metadata['height']}")
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to connect to stream: {e}")
            self.is_connected = False
            raise ConnectionError(f"Camera stream connection failed: {e}")

    def consume(self, **kwargs) -> Iterator[StreamData]:
        """
        Consumes frames from the connected video stream and yields them as NumPy arrays.
        
        Yields:
            StreamData (np.ndarray): The raw video frame.
        """
        if not self.video_capture:
            raise RuntimeError("Stream is not connected. Call connect() first.")
        
        while self.is_connected and self.video_capture.isOpened():
            # Read a frame
            ret, frame = self.video_capture.read()
            
            if not ret:
                logger.info(f"[{self.connector_id}] End of stream or failed to read frame.")
                break # Exit loop if stream ends or read fails

            # Convert BGR (OpenCV default) to RGB (ML standard)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yield the frame for processing
            yield rgb_frame

    def produce(self, frame: StreamData, destination_topic: Optional[str] = None, **kwargs):
        """
        Writes a processed frame (e.g., with annotations/bboxes) to a video file or stream output.

        Args:
            frame (np.ndarray): The processed frame (assumed RGB).
            destination_topic (Optional[str]): Placeholder, typically used to trigger video writer initialization.
        """
        if not self.output_path:
            logger.warning("Output path is not configured. Skipping frame writing.")
            return

        # 1. Initialize VideoWriter lazily (only on the first call to produce)
        if not self.video_writer:
            self._initialize_video_writer(frame.shape[1], frame.shape[0])

        if self.video_writer:
            # Convert frame back to BGR for OpenCV writer
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)
            logger.debug("Frame produced to output video.")

    def _initialize_video_writer(self, width: int, height: int):
        """Helper to set up the VideoWriter based on configuration."""
        if self.video_writer:
            return
            
        try:
            # Use XVID codec as default for cross-platform compatibility
            fourcc = cv2.VideoWriter_fourcc(*self.output_config.get('codec', 'XVID'))
            fps = self.output_config.get('fps', self.frame_metadata.get('fps', 30.0))
            
            self.video_writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                fps, 
                (width, height)
            )
            logger.info(f"VideoWriter initialized for output: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to initialize VideoWriter: {e}")
            self.video_writer = None # Ensure it remains None on failure

    def close(self):
        """
        Safely releases the VideoCapture and VideoWriter resources.
        CRITICAL for production stability.
        """
        if self.video_capture:
            self.video_capture.release()
            logger.info(f"[{self.connector_id}] VideoCapture released.")
        
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"[{self.connector_id}] VideoWriter released.")
            
        super().close()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are closed even if an exception occurs."""
        self.close()
        return super().__exit__(exc_type, exc_val, exc_tb)