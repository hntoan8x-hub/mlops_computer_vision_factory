# shared_libs/data_labeling/configs/label_schema.py

from pydantic import BaseModel, Field, conint, confloat, constr, validator
from typing import List, Dict, Any, Optional, Tuple

# Type Alias for Bounding Box: [x_min, y_min, x_max, y_max] normalized to [0, 1]
# Hardening: Use conlist and confloat for strict dimension and range checking.
BBoxNormalized = Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]

class BaseLabel(BaseModel):
    """Base model for all CV Labels, enforcing common metadata."""
    image_path: constr(min_length=5) = Field(..., description="URI or path to the source image.")
    timestamp_utc: Optional[str] = Field(None, description="Timestamp when the label was created/modified (ISO format).")
    labeler_source: constr(min_length=3) = Field("auto", description="Source of the label (auto, manual, semi).")

    @validator('image_path')
    def path_must_have_extension(cls, v):
        """Rule: Ensure the image path has a valid file extension."""
        if '.' not in v:
            raise ValueError("Image path must contain a file extension.")
        return v
    
    class Config:
        """Pydantic configuration."""
        # Forbid extra fields to prevent accidental injection of non-schema data
        extra = "forbid"

# --- Classification Label ---

class ClassificationLabel(BaseLabel):
    """Schema for Image Classification label."""
    label: constr(min_length=1) = Field(..., description="The predicted class name or ID.")

# --- Detection Label ---

class DetectionObject(BaseModel):
    """A single object detected within an image."""
    class_name: constr(min_length=1) = Field(..., description="Name of the detected class.")
    bbox: BBoxNormalized = Field(..., description="Bounding box [x_min, y_min, x_max, y_max] normalized to [0, 1].")
    confidence: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Confidence score (0.0 to 1.0).")
    
    @validator('bbox')
    def bbox_coordinates_valid(cls, v):
        """Rule: Ensure x_min < x_max and y_min < y_max."""
        x_min, y_min, x_max, y_max = v
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("BBox coordinates must satisfy x_min < x_max and y_min < y_max.")
        return v

class DetectionLabel(BaseLabel):
    """Schema for Object Detection label (list of objects)."""
    objects: List[DetectionObject] = Field(..., description="List of detected objects in the image.")
    
    @validator('objects')
    def objects_list_not_empty(cls, v):
        """Rule: The objects list cannot be empty for a valid detection label."""
        if not v:
            raise ValueError("Detection 'objects' list cannot be empty.")
        return v

# --- Segmentation Label ---

class SegmentationLabel(BaseLabel):
    """Schema for Instance/Semantic Segmentation label."""
    # Chúng ta lưu đường dẫn đến mask file (PNG/RLE/COCO format) thay vì array thô
    mask_path: constr(min_length=5) = Field(..., description="Local/Cloud path to the generated segmentation mask file (e.g., PNG).")
    class_name: Optional[constr(min_length=1)] = Field(None, description="Class name of the main object in the mask.")

# --- OCR Label ---

class OCRToken(BaseModel):
    """A single text token (word or character) extracted by OCR."""
    text: constr(min_length=1) = Field(..., description="The extracted text string.")
    bbox: BBoxNormalized = Field(..., description="BBox of the token normalized to [0, 1].")
    
class OCRLabel(BaseLabel):
    """Schema for Optical Character Recognition (OCR) label."""
    full_text: constr(min_length=1) = Field(..., description="The complete concatenated text recognized in the image.")
    tokens: List[OCRToken] = Field(..., description="List of individual tokens (words/characters) with their BBoxes.")
    
    @validator('tokens')
    def tokens_not_empty(cls, v):
        """Rule: The tokens list cannot be empty."""
        if not v:
            raise ValueError("OCR 'tokens' list cannot be empty.")
        return v

# --- Embedding Label ---

class EmbeddingLabel(BaseLabel):
    """Schema for Feature Vector Embedding label (used in retrieval/clustering)."""
    target_id: constr(min_length=1) = Field(..., description="ID of the entity/cluster this image belongs to.")
    # Lưu vector dưới dạng List[float] để dễ dàng serialize/deserialize
    vector: List[confloat()] = Field(..., description="The raw feature vector (embedding).")
    
    @validator('vector')
    def vector_dimension_check(cls, v):
        """Rule: Vector must have a dimension > 0."""
        if not v or len(v) < 1:
            raise ValueError("Embedding vector must have at least one dimension.")
        return v

# Định nghĩa kiểu Output chuẩn hóa cho tầng này (Giữ nguyên)
StandardLabel = Union[ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel, EmbeddingLabel]