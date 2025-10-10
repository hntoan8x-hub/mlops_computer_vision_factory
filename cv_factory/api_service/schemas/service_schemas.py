# cv_factory/api_service/schemas/service_schemas.py

# cv_factory/api_service/schemas/service_schemas.py (HARDENED OUTPUT)

from pydantic import BaseModel, Field, constr
from typing import List, Dict, Any, Union, Literal, Optional

# --- 1. Base Output Schemas (Can be simplified to reflect the Domain Entity) ---

# NOTE: Since the Postprocessor returns a structured Domain Entity (FinalDiagnosis), 
# we should align the API output directly with that Domain Entity structure 
# to ensure consistency across the whole system.
# The original structure (ClassificationPrediction, BoundingBox, etc.) is too granular 
# for the final API response.

# Using the structure of FinalDiagnosis as the basis for the API output contract:
class FinalPredictionSummary(BaseModel):
    """Reflects the FinalDiagnosis entity from the domain layer."""
    primary_finding: str = Field(..., description="The main finding or diagnosis (e.g., 'Pneumonia').")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence score from the domain logic.")
    is_critical: bool = Field(False, description="True if the finding requires immediate clinical attention.")

class PredictionOutput(BaseModel):
    """
    The standardized output structure for the inference service response.
    """
    image_id: str = Field(..., description="The unique identifier of the image.")
    patient_id: str = Field(..., description="The patient's identifier.")
    
    # The summary result from the domain postprocessor
    diagnosis_summary: FinalPredictionSummary 

    processed_latency_ms: float = Field(..., description="Total processing time in milliseconds.")
    model_version: str = Field(..., description="Version of the model used for prediction.")
    
    additional_metadata: Optional[Dict[str, Any]] = Field({}, description="Any extra metadata from the model.")

# --- 2. Input Schema ---

class PredictionInput(BaseModel):
    """Schema for the input request body to the /predict route."""
    
    # Image data encoded in Base64 (Standard practice for API transmission)
    image_base64: constr(min_length=10) = Field(..., description="Image file encoded as a Base64 string.")
    
    # Optional parameters for the prediction logic
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for filtering results.")
    metadata: Dict[str, Any] = Field({}, description="Optional run metadata (e.g., source device ID).")

# --- 3. Error/Health Schemas ---

class HealthCheck(BaseModel):
    """Schema for the /health route."""
    status: Literal["ok", "degraded", "error"] = Field("ok", description="Service status.")
    model_endpoint: str = Field(..., description="The currently configured model endpoint.")

class ErrorResponse(BaseModel):
    """Standardized schema for API error responses."""
    detail: str
    error_code: int = 500