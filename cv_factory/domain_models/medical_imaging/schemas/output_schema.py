from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class ClassificationPrediction(BaseModel):
    class_name: str
    confidence: float

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_name: str
    confidence: float

class SegmentationMask(BaseModel):
    # This can be a flattened list or a URL to the mask file
    mask_data: str = Field(..., description="URL or string representation of the segmentation mask.")
    class_name: str
    confidence: float

class PredictionOutput(BaseModel):
    """
    Schema for the final prediction output, tailored for various CV tasks.
    """
    image_id: str = Field(..., description="The unique identifier of the image.")
    patient_id: str = Field(..., description="The patient's identifier.")
    prediction_time_ms: float = Field(..., description="Time taken for inference in milliseconds.")
    classification: Optional[ClassificationPrediction] = None
    detections: Optional[List[BoundingBox]] = None
    segmentation: Optional[List[SegmentationMask]] = None
    additional_metadata: Optional[Dict[str, Any]] = Field({}, description="Any extra metadata from the model.")