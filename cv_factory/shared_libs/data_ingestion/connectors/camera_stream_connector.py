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
    
    Attributes:
        source: Camera index (int) or video file path (str).
        output_path: Path to save the processed output video.
        output_config: Configuration for cv2.VideoWriter (e.g., codec, fps).
        video_capture: OpenCV VideoCapture object for input.
        video_writer: OpenCV VideoWriter object for output.
        frame_metadata: Dictionary to store stream properties (FPS, resolution).
    """

    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CameraStreamConnector.

        Args:
            connector_id: Unique ID.
            config: Configuration including 'source' (int/str) and 'output_path'/'output_config'.
        """
        super().__init__(connector_id, config)
        # Using .get() for safe access to config items
        self.source = self.config.get('source')  
        self.output_path = self.config.get('output_path')
        self.output_config = self.config.get('output_config', {})
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_metadata: Dict[str, Any] = {} 
        
    def connect(self) -> bool:
        """
        Initializes the OpenCV VideoCapture object for the input stream.

        Returns:
            bool: True if the connection is successful.

        Raises:
            ValueError: If the camera source is not specified.
            ConnectionError: If the video stream cannot be opened.
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

    def consume(self, frame_timeout_s: float = 5.0, **kwargs) -> Iterator[StreamData]:
        """
        Consumes frames from the connected video stream and yields them as NumPy arrays.
        
        Implements a timeout mechanism to detect stream hangs (Robustness Hardening).

        Args:
            frame_timeout_s: Maximum time (in seconds) to wait without receiving a new frame before breaking the loop.
            **kwargs: Optional custom parameters.
        
        Yields:
            StreamData: The raw video frame (RGB NumPy array).

        Raises:
            RuntimeError: If the stream is not connected.
        """
        if not self.video_capture:
            raise RuntimeError("Stream is not connected. Call connect() first.")
        
        start_time = time.time() # Track time of last successful frame read
        
        while self.is_connected and self.video_capture.isOpened():
            # Read a frame
            ret, frame = self.video_capture.read()
            
            if not ret:
                # If reading failed:
                elapsed = time.time() - start_time
                if elapsed > frame_timeout_s:
                    logger.error(f"[{self.connector_id}] Stream hung/stalled for {frame_timeout_s}s. Forcing disconnect.")
                    self.is_connected = False # Force disconnection
                    break 
                
                # If stream is from a file, it's the end. If from camera, retry briefly.
                is_file = isinstance(self.source, str) and (os.path.exists(self.source) or self.source.startswith(('http', 'rtsp')))
                if is_file and not self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) == self.video_capture.get(cv2.CAP_PROP_POS_FRAMES):
                    # For files, if we haven't reached the end, something is wrong.
                    logger.warning(f"[{self.connector_id}] Failed to read frame, possibly corrupted file or temporary error.")
                elif not is_file:
                    logger.debug(f"[{self.connector_id}] No frame received, retrying...")
                    time.sleep(0.01) # Small pause for streaming sources
                    continue # Try again

                # Break if end of file or connection forced break
                break 

            # Reset hang timer on successful read (Reliability)
            start_time = time.time() 

            # Convert BGR (OpenCV default) to RGB (ML standard)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yield the frame for processing
            yield rgb_frame

    def produce(self, frame: StreamData, destination_topic: Optional[str] = None, **kwargs):
        """
        Writes a processed frame (e.g., with annotations/bboxes) to a video file or stream output.

        Args:
            frame: The processed frame (assumed RGB NumPy array).
            destination_topic: Placeholder for stream output, unused here as it's a file writer.
            **kwargs: Optional configuration (e.g., output_config overrides).
        """
        if not self.output_path:
            logger.warning("Output path is not configured. Skipping frame writing.")
            return

        # 1. Initialize VideoWriter lazily (only on the first call to produce)
        # Assuming frame is a NumPy array:
        if isinstance(frame, np.ndarray) and not self.video_writer:
            self._initialize_video_writer(frame.shape[1], frame.shape[0])

        if self.video_writer and isinstance(frame, np.ndarray):
            # Convert frame back to BGR for OpenCV writer
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)
            logger.debug("Frame produced to output video.")
        elif not isinstance(frame, np.ndarray):
             logger.error("Produce received non-NumPy array data, skipping video write.")


    def _initialize_video_writer(self, width: int, height: int):
        """
        Helper to set up the VideoWriter based on configuration.
        
        Args:
            width: Width of the frames.
            height: Height of the frames.
        """
        if self.video_writer:
            return
            
        try:
            # Use XVID codec as default for cross-platform compatibility
            codec = self.output_config.get('codec', 'XVID')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            fps = self.output_config.get('fps', self.frame_metadata.get('fps', 30.0))
            
            self.video_writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                fps, 
                (width, height)
            )
            # Hardening: Check if the writer successfully opened
            if not self.video_writer.isOpened():
                 raise IOError(f"VideoWriter failed to open file with codec {codec}.")
                 
            logger.info(f"VideoWriter initialized for output: {self.output_path} (Codec: {codec}, FPS: {fps})")
        except Exception as e:
            logger.error(f"Failed to initialize VideoWriter: {e}")
            self.video_writer = None 

    def close(self):
        """
        Safely releases the VideoCapture and VideoWriter resources.
        CRITICAL for production stability.
        """
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            logger.info(f"[{self.connector_id}] VideoCapture released.")
        
        # Hardening: Explicitly check if writer is open before releasing
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            logger.info(f"[{self.connector_id}] VideoWriter released.")
            
        super().close()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensure resources are closed even if an exception occurs via Context Manager.
        """
        return super().__exit__(exc_type, exc_val, exc_tb)