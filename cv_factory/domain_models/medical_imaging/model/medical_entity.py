# domain_models/medical_imaging/model/medical_entity.py

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class FinalDiagnosis(BaseModel):
    """
    Represents the final, human-readable medical diagnosis outcome.
    This is the output of the MedicalPostprocessor.
    """
    patient_id: str
    image_id: str
    primary_finding: str = Field(..., description="The main finding or diagnosis (e.g., 'Pneumonia', 'Nodule').")
    confidence_level: float = Field(..., description="Confidence score from the domain logic after thresholding.")
    is_critical: bool = Field(False, description="True if the finding requires immediate clinical attention.")
    raw_prediction_metadata: Dict[str, Any] = Field({}, description="Metadata from the raw model output.")

    def to_json(self) -> Dict[str, Any]:
        """Converts the entity to a dictionary suitable for API response or database storage."""
        return self.dict()