# cv_factory/shared_libs/data_ingestion/connectors/image_connector.py

import os
import io
import logging
from typing import Dict, Any, List, Union, Optional
from urllib.parse import urlparse

# Cloud Libraries
import boto3
from google.cloud import storage
from botocore.exceptions import ClientError as S3ClientError

# Image Processing
from PIL import Image
import numpy as np

# Import new Base Abstraction
# NOTE: Update the import path based on the new structure
from ..base.base_data_connector import BaseDataConnector, OutputData 
# Assuming BaseDataConnector's OutputData is Union[np.ndarray, List[np.ndarray], ...]

# Optional: Import cv2 only if needed, keeping PIL as primary
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


logger = logging.getLogger(__name__)

class ImageConnector(BaseDataConnector):
    """
    Concrete connector for handling image data from local paths, S3, and GCS.
    
    Implements the BaseDataConnector contract with read (load) and write (save) capabilities.
    """
    
    # --- Initialization and Connection Management (connect, close) ---

    def __init__(self, connector_id: str, **kwargs: Dict[str, Any]):
        """
        Initializes the ImageConnector with optional client configurations.
        """
        super().__init__(connector_id, kwargs)
        self.s3_client = None
        self.gcs_client = None
        self.boto3_config = self.config.get("boto3_config", {})
        self.gcs_config = self.config.get("gcs_config", {})
        self.use_pillow = self.config.get("use_pillow", True)
        self.is_connected = False

    def connect(self) -> bool:
        """
        Initializes cloud clients (S3, GCS) if their configs are present.
        Lazy loading of clients is kept for efficiency.
        """
        if self.is_connected:
            return True
            
        try:
            # S3 Client Initialization (Lazy Load Logic is moved out of connect)
            if self.config.get("aws_enabled", True): # Assume enabled by default
                self._get_s3_client()
                
            # GCS Client Initialization
            if self.config.get("gcp_enabled", True): # Assume enabled by default
                self._get_gcs_client()
            
            self.is_connected = True
            logger.info(f"[{self.connector_id}] ImageConnector initialized.")
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to connect/initialize cloud clients: {e}")
            self.is_connected = False
            raise ConnectionError(f"Connector initialization failed: {e}")

    def close(self):
        """
        Closes cloud clients and releases resources.
        """
        if self.s3_client:
            # Boto3 clients often don't require explicit close, but we log it.
            self.s3_client = None 
        if self.gcs_client:
            # GCS clients usually handle resource release automatically.
            self.gcs_client = None 
            
        super().close() # Calls the BaseDataConnector close log

    def _get_s3_client(self):
        """Lazy-loads and returns the S3 client."""
        if self.s3_client is None:
            self.s3_client = boto3.client("s3", **self.boto3_config)
            logger.info(f"[{self.connector_id}] S3 client initialized.")
        return self.s3_client

    def _get_gcs_client(self):
        """Lazy-loads and returns the GCS client."""
        if self.gcs_client is None:
            self.gcs_client = storage.Client(**self.gcs_config)
            logger.info(f"[{self.connector_id}] GCS client initialized.")
        return self.gcs_client

    # --- Read/Load Logic (read) ---

    def _load_bytes_to_array(self, img_data: bytes) -> np.ndarray:
        """Converts raw image bytes into a NumPy array using PIL."""
        try:
            img_stream = io.BytesIO(img_data)
            with Image.open(img_stream) as img:
                return np.array(img.convert("RGB"))
        except Exception as e:
            raise IOError(f"Failed to decode image bytes using PIL: {e}")

    def _read_from_local_file(self, file_path: str) -> np.ndarray:
        """Loads a single image from a local file path."""
        if self.use_pillow:
            with Image.open(file_path) as img:
                return np.array(img.convert("RGB"))
        elif OPENCV_AVAILABLE:
            img = cv2.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"OpenCV could not read the image at {file_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
             # Fallback if PIL is disabled and OpenCV isn't available
             with open(file_path, 'rb') as f:
                 return self._load_bytes_to_array(f.read())
        
    def _read_from_s3(self, s3_url: str) -> np.ndarray:
        """Loads a single image from an S3 URL."""
        client = self._get_s3_client()
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")

        try:
            response = client.get_object(Bucket=bucket_name, Key=object_key)
            img_data = response['Body'].read()
            return self._load_bytes_to_array(img_data)
        except S3ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: {s3_url}")
            raise
        except Exception as e:
            raise IOError(f"Error loading image from S3 '{s3_url}': {e}")

    def _read_from_gcs(self, gcs_url: str) -> np.ndarray:
        """Loads a single image from a GCS URL."""
        client = self._get_gcs_client()
        parsed_url = urlparse(gcs_url)
        bucket_name = parsed_url.netloc
        blob_name = parsed_url.path.lstrip("/")

        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            img_data = blob.download_as_bytes()
            return self._load_bytes_to_array(img_data)
        except Exception as e:
            raise IOError(f"Error loading image from GCS '{gcs_url}': {e}")

    def read(self, source_uri: str, **kwargs: Dict[str, Any]) -> OutputData:
        """
        Reads image data from the specified source_uri.
        
        If source_uri points to a single file, returns np.ndarray.
        If source_uri points to a directory/bucket (and kwargs['recursive'] is True), 
        returns List[np.ndarray].
        
        Args:
            source_uri (str): Path to a single image file or a directory/bucket prefix.
            **kwargs: Extra parameters (e.g., recursive=True).
            
        Returns:
            OutputData: A single image array or a list of image arrays.
        """
        if not self.is_connected:
            self.connect() # Ensure connection is established
            
        is_directory = kwargs.get("recursive", False)

        if os.path.isfile(source_uri) or not source_uri.startswith(("s3://", "gs://")) and not is_directory:
            return self._read_from_local_file(source_uri)
        elif source_uri.startswith("s3://"):
            if is_directory:
                return self._read_from_s3_folder(source_uri, **kwargs)
            return self._read_from_s3(source_uri)
        elif source_uri.startswith("gs://"):
            if is_directory:
                return self._read_from_gcs_folder(source_uri, **kwargs)
            return self._read_from_gcs(source_uri)
        elif os.path.isdir(source_uri) and is_directory:
            return self._read_from_local_folder(source_uri, **kwargs)
        else:
            raise ValueError(f"Unsupported source URI or invalid recursive flag: {source_uri}")

    # --- Folder/Recursive Read Logic (renamed from _load_from_local_folder) ---
    # These private methods handle recursive loading for read()

    def _read_from_local_folder(self, folder_path: str, **kwargs) -> List[np.ndarray]:
        """Loads all images from a local folder recursively."""
        images = []
        supported_exts = kwargs.get("supported_exts", [".jpg", ".jpeg", ".png", ".bmp"])
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_exts):
                    file_path = os.path.join(root, file)
                    try:
                        images.append(self._read_from_local_file(file_path))
                    except Exception as e:
                        logger.warning(f"[{self.connector_id}] Skipping file '{file_path}' due to read error: {e}")
        return images
        
    def _read_from_s3_folder(self, s3_prefix: str, **kwargs) -> List[np.ndarray]:
        """Loads all images under an S3 prefix (simulated folder)."""
        # NOTE: Implement actual listing logic using S3 client here.
        logger.warning(f"[{self.connector_id}] S3 folder read is not fully implemented: {s3_prefix}")
        return []

    def _read_from_gcs_folder(self, gcs_prefix: str, **kwargs) -> List[np.ndarray]:
        """Loads all images under a GCS prefix (simulated folder)."""
        # NOTE: Implement actual listing logic using GCS client here.
        logger.warning(f"[{self.connector_id}] GCS folder read is not fully implemented: {gcs_prefix}")
        return []

    # --- Write/Persist Logic (write) ---

    def _write_to_s3(self, data: np.ndarray, destination_url: str, file_format: str) -> str:
        """Writes an image array to S3."""
        client = self._get_s3_client()
        parsed_url = urlparse(destination_url)
        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        
        # Ensure object_key has the correct extension
        final_key = f"{object_key}.{file_format}" if not object_key.lower().endswith(f".{file_format}") else object_key

        try:
            # Save array to bytes stream
            img = Image.fromarray(data)
            img_stream = io.BytesIO()
            img.save(img_stream, format=file_format.upper()) # PIL requires upper-case format
            img_stream.seek(0)

            client.put_object(
                Bucket=bucket_name, 
                Key=final_key, 
                Body=img_stream,
                ContentType=f'image/{file_format}'
            )
            return f"s3://{bucket_name}/{final_key}"
        except Exception as e:
            raise IOError(f"Error writing image to S3 '{destination_url}': {e}")
            
    def _write_to_gcs(self, data: np.ndarray, destination_url: str, file_format: str) -> str:
        """Writes an image array to GCS."""
        client = self._get_gcs_client()
        parsed_url = urlparse(destination_url)
        bucket_name = parsed_url.netloc
        blob_name = parsed_url.path.lstrip("/")
        
        final_name = f"{blob_name}.{file_format}" if not blob_name.lower().endswith(f".{file_format}") else blob_name

        try:
            # Save array to bytes stream
            img = Image.fromarray(data)
            img_stream = io.BytesIO()
            img.save(img_stream, format=file_format.upper())
            img_stream.seek(0)
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(final_name)
            blob.upload_from_file(img_stream, content_type=f'image/{file_format}')
            
            return f"gs://{bucket_name}/{final_name}"
        except Exception as e:
            raise IOError(f"Error writing image to GCS '{destination_url}': {e}")

    def _write_to_local(self, data: np.ndarray, destination_path: str, file_format: str) -> str:
        """Writes an image array to a local file."""
        full_path = f"{destination_path}.{file_format}" if not destination_path.lower().endswith(f".{file_format}") else destination_path

        try:
            img = Image.fromarray(data)
            img.save(full_path, format=file_format.upper())
            return full_path
        except Exception as e:
            raise IOError(f"Error writing image to local path '{full_path}': {e}")

    def write(self, data: OutputData, destination_uri: str, file_format: str = 'png', **kwargs) -> str:
        """
        Writes/Persists a single image (NumPy array) to the specified destination.
        
        NOTE: This implementation currently only supports writing a single np.ndarray.
        Writing a list of arrays would require iteration and should be handled externally 
        or in a separate high-level orchestrator for clarity.
        
        Args:
            data (np.ndarray): The image data to save (H, W, C).
            destination_uri (str): The local path or cloud URI prefix.
            file_format (str): Desired file extension (e.g., 'png', 'jpg').
            
        Returns:
            str: The final URI/path of the saved image.
            
        Raises:
            TypeError: If data is not a NumPy array.
            ValueError: For unsupported URI schemes.
        """
        if not self.is_connected:
            self.connect()
            
        if not isinstance(data, np.ndarray):
            raise TypeError("ImageConnector 'write' method only accepts a single numpy.ndarray.")
        
        if destination_uri.startswith("s3://"):
            return self._write_to_s3(data, destination_uri, file_format)
        elif destination_uri.startswith("gs://"):
            return self._write_to_gcs(data, destination_uri, file_format)
        else:
            return self._write_to_local(data, destination_uri, file_format)