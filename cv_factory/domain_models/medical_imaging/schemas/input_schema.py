from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, conbytes

class MedicalImageInput(BaseModel):
    """
    Schema for validating raw medical image input data.

    This ensures that incoming data has the correct structure and includes
    essential metadata for processing.
    """
    image_id: str = Field(..., description="A unique identifier for the image.")
    patient_id: str = Field(..., description="The patient's identifier.")
    modality: str = Field(..., description="The imaging modality (e.g., 'CT', 'MRI', 'X-ray').")
    image_data: conbytes(min_length=1, max_length=100000000) = Field(..., description="The raw image data as bytes.")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional key-value metadata.")