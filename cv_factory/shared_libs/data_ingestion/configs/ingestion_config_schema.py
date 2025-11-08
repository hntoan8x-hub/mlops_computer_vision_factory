# cv_factory/shared_libs/data_ingestion/configs/ingestion_config_schema.py

from typing import List, Literal, Dict, Any, Optional, Union
from urllib.parse import urlparse # Required for custom URI validation
from pydantic import (
    BaseModel, 
    Field, 
    conint, 
    constr, 
    validator, 
    root_validator,
    AnyUrl, # For generic URLs (s3, gs, http, file)
    HttpUrl, # For strict HTTP/HTTPS URLs
    NonNegativeInt, # For non-negative integers (e.g., timeout)
    conlist # For enforcing list constraints (e.g., min length)
)

# --- 1. Base and Atomic Connector Schemas (The Rules) ---

# Define Supported Connector Types using Literal for type safety
CONNECTOR_TYPES = Literal["image", "video", "dicom", "api", "kafka", "camera"]

class BaseConnectorConfig(BaseModel):
    """
    Base model for all data connector configurations.

    Attributes:
        type: The type of connector (e.g., 'image', 'kafka', 'dicom').
        uri: The base URI or path to the source data (e.g., s3://bucket/path/ or /local/path/).
    """
    type: CONNECTOR_TYPES = Field(..., description="The type of connector (e.g., 'image', 'kafka', 'dicom').")
    uri: constr(min_length=5) = Field(..., description="The base URI or path to the source data (e.g., s3://bucket/path/ or /local/path/).")
    
    class Config:
        """Pydantic configuration."""
        # Allow passing extra params that the specific connector might use
        extra = "allow" 

# --- 1.1 Specific Connector Schemas (Enforcing Specific Rules) ---

class DICOMConnectorConfig(BaseConnectorConfig):
    """
    Configuration rules specific to DICOM data handling.

    Attributes:
        type: Must be "dicom".
        anonymize_phi: If True, scrub PHI (Protected Health Information) during ingestion.
    """
    type: Literal["dicom"] = "dicom" # Enforce type for this schema
    anonymize_phi: bool = Field(False, description="If True, scrub PHI (Protected Health Information) during ingestion.")
    
    @validator('uri')
    def check_dicom_uri_scheme(cls, v: str) -> str:
        """
        Rule: Restrict DICOM URIs to local paths, S3, or GCS for security compliance.

        Args:
            v: The input URI string.

        Returns:
            The validated URI string.

        Raises:
            ValueError: If the URI scheme is not supported for DICOM data.
        """
        parsed = urlparse(v)
        # Allow empty scheme (local path), 'file', 's3', or 'gs'
        if parsed.scheme not in ['', 'file', 's3', 'gs']: 
            raise ValueError("DICOM URI must be a local path, S3, or GCS for security compliance.")
        return v
    
class KafkaConnectorConfig(BaseConnectorConfig):
    """
    Configuration rules specific to Kafka streaming.

    Attributes:
        type: Must be "kafka".
        bootstrap_servers: List of Kafka broker addresses (must not be empty).
        topic: The Kafka topic to consume from.
        group_id: Consumer group ID for distributed processing.
    """
    type: Literal["kafka"] = "kafka"
    # Hardening: Use conlist for non-empty check
    bootstrap_servers: conlist(str, min_length=1) = Field(..., description="List of Kafka broker addresses.") 
    topic: constr(min_length=1) = Field(..., description="The Kafka topic to consume from.")
    group_id: constr(min_length=1) = Field(..., description="Consumer group ID for distributed processing.")
    
    # Removed explicit validator as conlist is stronger and clearer.
        
class CameraConnectorConfig(BaseConnectorConfig):
    """
    Configuration rules specific to local or RTSP camera streams.

    Attributes:
        type: Must be "camera".
        fps_limit: Maximum frames per second to process (1 to 60).
    """
    type: Literal["camera"] = "camera"
    fps_limit: conint(gt=0, le=60) = Field(30, description="Maximum frames per second to process.")
    
    
class ImageConnectorConfig(BaseConnectorConfig):
    """
    Configuration rules specific to Image file handling (Local/S3/GCS).

    Attributes:
        type: Must be "image".
        uri: URI to the image source (must be a valid URL/Path).
        supported_formats: List of supported file extensions.
    """
    type: Literal["image"] = "image"
    # Using AnyUrl allows protocols like s3://, gs://, file://
    uri: AnyUrl = Field(..., description="URI to the image source (must be a valid URL/Path).")
    supported_formats: List[str] = Field(['jpg', 'png', 'dicom'], description="List of supported file extensions.")

class VideoConnectorConfig(BaseConnectorConfig):
    """
    Configuration rules specific to Video file handling (static video files).

    Attributes:
        type: Must be "video".
        uri: URI to the video file.
        load_type: Load video as full frames or just metadata.
    """
    type: Literal["video"] = "video"
    uri: AnyUrl = Field(..., description="URI to the video file.")
    load_type: Literal["full", "metadata_only"] = Field("full", description="Load video as full frames or just metadata.")

class APIConnectorConfig(BaseConnectorConfig):
    """
    Configuration rules specific to REST API connections.

    Attributes:
        type: Must be "api".
        uri: The base URL of the API endpoint (must be HTTP/HTTPS).
        auth_method: Authentication method (bearer, basic, none).
        timeout_seconds: Connection timeout (must be non-negative).
    """
    type: Literal["api"] = "api"
    # Hardening: Using HttpUrl ensures http:// or https:// scheme
    uri: HttpUrl = Field(..., description="The base URL of the API endpoint.")
    auth_method: Optional[Literal["bearer", "basic", "none"]] = Field("none", description="Authentication method.")
    timeout_seconds: NonNegativeInt = Field(10, description="Connection timeout.")

    # Removed explicit check_http_prefix validator as HttpUrl handles it internally.


# Use a Union of specific types for the main sources list
SourceConnector = Union[
    ImageConnectorConfig, 
    VideoConnectorConfig, 
    APIConnectorConfig,
    DICOMConnectorConfig, 
    KafkaConnectorConfig, 
    CameraConnectorConfig,
    BaseConnectorConfig # For generic types that haven't been specialized yet
]
# --- 2. Main Ingestion Configuration ---
class IngestionConfig(BaseModel):
    """
    Master schema for the Data Ingestion configuration.
    
    Uses Union types to enforce specific rules based on the 'type' field 
    for each connector in the list.

    Attributes:
        connectors: A list of data sources/connectors to initialize.
        data_split_ratio: Ratios for data splitting (e.g., train/validation/test).
        metadata_uri: URI to an external metadata manifest (optional).
    """
    
    # List must contain models conforming to the strict rules of each connector type
    connectors: List[SourceConnector] = Field(..., description="A list of data sources/connectors to initialize.")
    
    # Additional configuration settings
    data_split_ratio: Dict[str, float] = Field({"train": 0.8, "validation": 0.2}, description="Ratios for data splitting (e.g., train/validation/test).")
    metadata_uri: Optional[str] = Field(None, description="URI to an external metadata manifest (optional).")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        # Hardening: Forbid extra fields to catch typos in production configs
        extra = "forbid" 

    @root_validator(pre=True)
    def validate_unique_uris(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule: Ensure no two data sources share the exact same URI (path) to prevent conflict.

        Args:
            values: The input configuration dictionary.

        Returns:
            The validated input configuration dictionary.

        Raises:
            ValueError: If duplicate URIs are found.
        """
        uris = [c.get('uri') for c in values.get('connectors', []) if isinstance(c, dict) and c.get('uri')]
        if len(uris) != len(set(uris)):
            raise ValueError("All data source URIs must be unique to prevent conflict.")
        return values