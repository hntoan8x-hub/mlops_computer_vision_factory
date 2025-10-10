# cv_factory/api_service/api_config.py

import os
import logging
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- 1. Pydantic Schema for Configuration Validation ---

class ApiServiceConfig(BaseModel):
    """
    Schema for validating all critical service and endpoint configurations,
    typically loaded from environment variables.
    """
    
    # 1. Service Control
    service_host: str = Field("0.0.0.0", description="Host address for the FastAPI service.")
    service_port: int = Field(8000, gt=1023, description="Port for the FastAPI service to listen on.")
    
    # 2. Deployment Target (Where the model is actually running)
    cloud_provider: Literal["aws", "gcp", "azure"] = Field(
        ..., description="Target cloud platform for the deployed ML endpoint."
    )
    endpoint_url: str = Field(..., description="The invocation URL of the deployed SageMaker/Vertex/AML endpoint.")
    
    # 3. Model/Version Tracking
    model_version: str = Field("Production", description="The target model version/stage (e.g., 'v1.2' or 'Production').")
    
    # 4. Security (Loaded via Environment Variable Check)
    auth_token: Optional[str] = Field(None, description="Authentication token required for cloud endpoint invocation (if applicable).")

    # 5. Operational Settings
    log_level: str = Field("INFO", description="Logging level for the service.")
    
    class Config:
        # Pydantic will read environment variables for fields if they match (e.g., ENDPOINT_URL)
        env_prefix = 'CV_API_' 
        extra = "ignore" # Ignore extra env vars not defined here

# --- 2. Configuration Loader ---

def load_api_config() -> ApiServiceConfig:
    """
    Loads and validates the API service configuration from environment variables.
    
    Raises:
        ValueError: If critical environment variables are missing (e.g., ENDPOINT_URL).
    """
    # Load and validate the configuration
    try:
        # We manually construct the dictionary from environment variables 
        # to ensure critical fields are present.
        config_data = {
            "cloud_provider": os.environ.get("CLOUD_PROVIDER", "aws"), # Use a default
            "endpoint_url": os.environ.get("ENDPOINT_URL"),
            "service_port": os.environ.get("SERVICE_PORT"),
            "model_version": os.environ.get("MODEL_VERSION"),
            "auth_token": os.environ.get("AUTH_TOKEN"),
            "log_level": os.environ.get("LOG_LEVEL"),
        }
        
        validated_config = ApiServiceConfig(**{k: v for k, v in config_data.items() if v is not None})
        logger.info(f"API service config loaded successfully. Target: {validated_config.cloud_provider}")
        return validated_config
        
    except Exception as e:
        logger.critical(f"FATAL: Missing or invalid environment configuration for API Service: {e}")
        raise ValueError(f"Missing critical API configuration parameters: {e}")

# Export the validated config object for easy import
API_CONFIG = load_api_config()