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

# Import Base Abstraction
from ..base.base_data_connector import BaseDataConnector, OutputData 

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
    
    Implements the BaseDataConnector contract with read (load) and write (save) 
    capabilities, supporting single file and recursive directory/bucket loading.
    """
    
    # --- Initialization and Connection Management (connect, close) ---

    def __init__(self, connector_id: str, **kwargs: Dict[str, Any]):
        """
        Initializes the ImageConnector with optional client configurations.

        Args:
            connector_id: A unique identifier for this connector instance.
            **kwargs: Configuration dictionary, potentially containing cloud client settings.
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

        Returns:
            bool: True if connection/initialization is successful.

        Raises:
            ConnectionError: If client initialization fails.
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
            self.s3_client = None 
        if self.gcs_client:
            self.gcs_client = None 
            
        super().close()

    def _get_s3_client(self):
        """
        Lazy-loads and returns the S3 client, handling credentials implicitly.
        """
        if self.s3_client is None:
            # Hardening: Use a session for better resource management
            session = boto3.Session(**self.boto3_config)
            self.s3_client = session.client("s3")
            logger.info(f"[{self.connector_id}] S3 client initialized.")
        return self.s3_client

    def _get_gcs_client(self):
        """
        Lazy-loads and returns the GCS client, handling credentials implicitly.
        """
        if self.gcs_client is None:
            self.gcs_client = storage.Client(**self.gcs_config)
            logger.info(f"[{self.connector_id}] GCS client initialized.")
        return self.gcs_client

    # --- Read/Load Logic (read) ---

    def _load_bytes_to_array(self, img_data: bytes) -> np.ndarray:
        """
        Converts raw image bytes into a NumPy array using PIL.

        Args:
            img_data: The raw image file content in bytes.

        Returns:
            The image as a NumPy array (H, W, C), converted to RGB.

        Raises:
            IOError: If decoding fails.
        """
        try:
            img_stream = io.BytesIO(img_data)
            with Image.open(img_stream) as img:
                return np.array(img.convert("RGB"))
        except Exception as e:
            raise IOError(f"Failed to decode image bytes using PIL: {e}")

    def _read_from_local_file(self, file_path: str) -> np.ndarray:
        """
        Loads a single image from a local file path.

        Args:
            file_path: The local path to the image file.

        Returns:
            The image as a NumPy array.
        """
        if self.use_pillow:
            with Image.open(file_path) as img:
                return np.array(img.convert("RGB"))
        elif OPENCV_AVAILABLE:
            img = cv2.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"OpenCV could not read the image at {file_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
             with open(file_path, 'rb') as f:
                 return self._load_bytes_to_array(f.read())
        
    def _read_from_s3(self, s3_url: str) -> np.ndarray:
        """
        Loads a single image from an S3 URL (s3://bucket/key).

        Args:
            s3_url: The full S3 URI.

        Returns:
            The image as a NumPy array.

        Raises:
            FileNotFoundError: If the S3 key is not found.
        """
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
        """
        Loads a single image from a GCS URL (gs://bucket/blob).

        Args:
            gcs_url: The full GCS URI.

        Returns:
            The image as a NumPy array.
        """
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
        If source_uri points to a directory/bucket prefix (and kwargs['recursive'] is True), 
        returns List[np.ndarray].
        
        Args:
            source_uri: Path to a single image file or a directory/bucket prefix.
            **kwargs: Extra parameters (e.g., recursive=True, supported_exts).
            
        Returns:
            OutputData: A single image array or a list of image arrays.

        Raises:
            ValueError: For unsupported URI schemes or invalid flag combinations.
        """
        if not self.is_connected:
            self.connect() 
            
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

    # --- Folder/Recursive Read Logic (Implemented) ---

    def _read_from_local_folder(self, folder_path: str, **kwargs) -> List[np.ndarray]:
        """
        Loads all images from a local folder recursively.

        Args:
            folder_path: The root directory path.
            **kwargs: Contains supported_exts list.

        Returns:
            A list of image NumPy arrays.
        """
        images = []
        supported_exts = kwargs.get("supported_exts", [".jpg", ".jpeg", ".png", ".bmp"])
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Basic extension check (case insensitive)
                if any(file.lower().endswith(ext) for ext in supported_exts):
                    file_path = os.path.join(root, file)
                    try:
                        images.append(self._read_from_local_file(file_path))
                    except Exception as e:
                        logger.warning(f"[{self.connector_id}] Skipping local file '{file_path}' due to read error: {e}")
        return images
        
    def _read_from_s3_folder(self, s3_prefix: str, **kwargs) -> List[np.ndarray]:
        """
        Loads all images under an S3 prefix (simulated folder) recursively.

        Args:
            s3_prefix: The S3 URI prefix (e.g., s3://bucket/folder/).
            **kwargs: Contains supported_exts list.

        Returns:
            A list of image NumPy arrays.
        """
        client = self._get_s3_client()
        parsed_url = urlparse(s3_prefix)
        bucket_name = parsed_url.netloc
        # Ensure prefix ends with '/' for directory-like listing if it's a folder.
        prefix = parsed_url.path.lstrip("/")
        supported_exts = kwargs.get("supported_exts", [".jpg", ".jpeg", ".png", ".bmp"])

        images = []
        paginator = client.get_paginator('list_objects_v2')
        # Use Pagination for large buckets (Scalability Hardening)
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    object_key = obj['Key']
                    if any(object_key.lower().endswith(ext) for ext in supported_exts):
                        s3_url = f"s3://{bucket_name}/{object_key}"
                        try:
                            # Reuse single-file reading logic
                            images.append(self._read_from_s3(s3_url)) 
                        except Exception as e:
                            logger.warning(f"[{self.connector_id}] Skipping S3 file '{s3_url}' due to read error: {e}")
        
        return images

    def _read_from_gcs_folder(self, gcs_prefix: str, **kwargs) -> List[np.ndarray]:
        """
        Loads all images under a GCS prefix (simulated folder) recursively.

        Args:
            gcs_prefix: The GCS URI prefix (e.g., gs://bucket/folder/).
            **kwargs: Contains supported_exts list.

        Returns:
            A list of image NumPy arrays.
        """
        client = self._get_gcs_client()
        parsed_url = urlparse(gcs_prefix)
        bucket_name = parsed_url.netloc
        prefix = parsed_url.path.lstrip("/")
        supported_exts = kwargs.get("supported_exts", [".jpg", ".jpeg", ".png", ".bmp"])

        images = []
        bucket = client.bucket(bucket_name)
        # Use list_blobs with prefix (GCS Pagination is handled internally)
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if any(blob.name.lower().endswith(ext) for ext in supported_exts):
                gcs_url = f"gs://{bucket_name}/{blob.name}"
                try:
                    # Reuse single-file reading logic
                    images.append(self._read_from_gcs(gcs_url))
                except Exception as e:
                    logger.warning(f"[{self.connector_id}] Skipping GCS file '{gcs_url}' due to read error: {e}")

        return images

    # --- Write/Persist Logic (write) ---

    def _write_to_s3(self, data: np.ndarray, destination_url: str, file_format: str) -> str:
        """
        Writes an image array to S3.

        Args:
            data: The image NumPy array (H, W, C).
            destination_url: The S3 URI prefix (e.g., s3://bucket/path).
            file_format: Desired file extension (e.g., 'png', 'jpg').

        Returns:
            The final S3 URI of the saved image.
        """
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
            img.save(img_stream, format=file_format.upper())
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
        """
        Writes an image array to GCS.

        Args:
            data: The image NumPy array (H, W, C).
            destination_url: The GCS URI prefix (e.g., gs://bucket/path).
            file_format: Desired file extension (e.g., 'png', 'jpg').

        Returns:
            The final GCS URI of the saved image.
        """
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
        """
        Writes an image array to a local file.

        Args:
            data: The image NumPy array (H, W, C).
            destination_path: The local file path.
            file_format: Desired file extension.

        Returns:
            The full local path of the saved image.
        """
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
        
        Note: This method is designed for single image persistence. Batch writing 
        should typically be handled by a higher-level orchestration layer.
        
        Args:
            data: The image data to save (H, W, C).
            destination_uri: The local path or cloud URI prefix.
            file_format: Desired file extension (e.g., 'png', 'jpg').
            
        Returns:
            str: The final URI/path of the saved image.
            
        Raises:
            TypeError: If data is not a NumPy array.
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