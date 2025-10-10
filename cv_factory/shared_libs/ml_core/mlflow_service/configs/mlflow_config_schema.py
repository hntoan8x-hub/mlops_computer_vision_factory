from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class TrackerConfig(BaseModel):
    type: str = Field("mlflow", description="The type of the tracker service.")
    params: Dict[str, Any] = Field({}, description="Parameters for the tracker.")

    @validator('type')
    def validate_tracker_type(cls, v):
        allowed_types = {"mlflow"}
        if v.lower() not in allowed_types:
            raise ValueError(f"Invalid tracker type: '{v}'. Supported types are: {allowed_types}")
        return v

class RegistryConfig(BaseModel):
    type: str = Field("mlflow", description="The type of the model registry service.")
    params: Dict[str, Any] = Field({}, description="Parameters for the registry.")

    @validator('type')
    def validate_registry_type(cls, v):
        allowed_types = {"mlflow"}
        if v.lower() not in allowed_types:
            raise ValueError(f"Invalid registry type: '{v}'. Supported types are: {allowed_types}")
        return v

class MLflowConfig(BaseModel):
    tracking_uri: Optional[str] = Field(None, description="The URI of the MLflow tracking server.")
    tracker: TrackerConfig
    registry: RegistryConfig