# cv_factory/shared_libs/data_ingestion/configs/ingestion_config_schema.py

from typing import List, Literal, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, conint, constr, validator, root_validator

# --- 1. Base and Atomic Connector Schemas (The Rules) ---

# Define Supported Connector Types using Literal for type safety
CONNECTOR_TYPES = Literal["image", "video", "dicom", "api", "kafka", "camera"]

class BaseConnectorConfig(BaseModel):
    """Base model for all data connector configurations."""
    type: CONNECTOR_TYPES = Field(..., description="The type of connector (e.g., 'image', 'kafka', 'dicom').")
    uri: constr(min_length=5) = Field(..., description="The base URI or path to the source data (e.g., s3://bucket/path/ or /local/path/).")
    
    class Config:
        # Allow passing extra params that the specific connector might use
        extra = "allow" 

# --- 1.1 Specific Connector Schemas (Enforcing Specific Rules) ---

class DICOMConnectorConfig(BaseConnectorConfig):
    """Rules specific to DICOM data handling."""
    type: Literal["dicom"] = "dicom" # Enforce type for this schema
    anonymize_phi: bool = Field(False, description="If True, scrub PHI (Protected Health Information) during ingestion.")
    
class KafkaConnectorConfig(BaseConnectorConfig):
    """Rules specific to Kafka streaming."""
    type: Literal["kafka"] = "kafka"
    bootstrap_servers: List[str] = Field(..., description="List of Kafka broker addresses.")
    topic: constr(min_length=1) = Field(..., description="The Kafka topic to consume from.")
    group_id: constr(min_length=1) = Field(..., description="Consumer group ID for distributed processing.")
    
    @validator('bootstrap_servers')
    def check_non_empty_servers(cls, v):
        if not v:
            raise ValueError("Bootstrap servers list cannot be empty for Kafka.")
        return v
        
class CameraConnectorConfig(BaseConnectorConfig):
    """Rules specific to local or RTSP camera streams."""
    type: Literal["camera"] = "camera"
    fps_limit: conint(gt=0, le=60) = Field(30, description="Maximum frames per second to process.")
    
    
class ImageConnectorConfig(BaseConnectorConfig):
    """Rules specific to Image file handling (Local/S3/GCS)."""
    type: Literal["image"] = "image"
    uri: AnyUrl = Field(..., description="URI to the image source (must be a valid URL/Path).")
    supported_formats: List[str] = Field(['jpg', 'png', 'dicom'], description="List of supported file extensions.")

class VideoConnectorConfig(BaseConnectorConfig):
    """Rules specific to Video file handling (static video files)."""
    type: Literal["video"] = "video"
    uri: AnyUrl = Field(..., description="URI to the video file.")
    load_type: Literal["full", "metadata_only"] = Field("full", description="Load video as full frames or just metadata.")

class APIConnectorConfig(BaseConnectorConfig):
    """Rules specific to REST API connections."""
    type: Literal["api"] = "api"
    uri: HttpUrl = Field(..., description="The base URL of the API endpoint.")
    auth_method: Optional[Literal["bearer", "basic", "none"]] = Field("none", description="Authentication method.")
    timeout_seconds: NonNegativeInt = Field(10, description="Connection timeout.")

    @validator('uri')
    def check_http_prefix(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("API URI must include http:// or https:// scheme.")
        return v


# Use a Union of specific types for the main sources list
SourceConnector = Union[
    ImageConnectorConfig, 
    VideoConnectorConfig, 
    APIConnectorConfig,
    DICOMConnectorConfig, 
    KafkaConnectorConfig, 
    CameraConnectorConfig,
    BaseConnectorConfig # For generic types
]
# --- 2. Main Ingestion Configuration ---
# (IngestionConfig uses the updated SourceConnector Union)
class IngestionConfig(BaseModel):
    """
    Master schema for the Data Ingestion configuration.
    Uses Union types to enforce specific rules based on the 'type' field.
    """
    
    # List must contain models conforming to the strict rules of each connector type
    connectors: List[SourceConnector] = Field(..., description="A list of data sources/connectors to initialize.")
    
    # Additional configuration settings
    data_split_ratio: Dict[str, float] = Field({"train": 0.8, "validation": 0.2}, description="Ratios for data splitting (e.g., train/validation/test).")
    metadata_uri: Optional[str] = Field(None, description="URI to an external metadata manifest (optional).")

    class Config:
        use_enum_values = True
        extra = "forbid" # For production, forbid extra fields to catch typos

    @root_validator(pre=True)
    def validate_unique_uris(cls, values):
        """Rule: Ensure no two data sources share the exact same URI (path)."""
        uris = [c.get('uri') for c in values.get('connectors', [])]
        if len(uris) != len(set(uris)):
            raise ValueError("All data source URIs must be unique to prevent conflict.")
        return values