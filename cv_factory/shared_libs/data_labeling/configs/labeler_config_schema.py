# shared_libs/data_labeling/configs/labeler_config_schema.py

from pydantic import BaseModel, Field, conint, confloat, constr, validator, root_validator
from typing import Dict, Any, List, Optional, Literal, Union

# --- 1. Sub-Schemas: Detailed configuration for each Labeler type ---

class ClassificationLabelerConfig(BaseModel):
    """
    Configuration for ClassificationLabeler.

    Attributes:
        label_source_uri: URI/path of the label file (e.g., labels.csv).
        image_path_column: Column name containing image paths in the label source.
        label_column: Column name containing the raw label (string or int).
        class_map_path: Path to a JSON file mapping label_name -> integer ID (optional).
    """
    label_source_uri: str = Field(..., description="URI/path of the label file (e.g., labels.csv).")
    image_path_column: str = Field("image_path", description="Column name containing image paths.")
    label_column: str = Field("label", description="Column name containing the raw label.")
    class_map_path: Optional[str] = Field(None, description="Path to a JSON file mapping label_name -> id.")

class DetectionLabelerConfig(BaseModel):
    """
    Configuration for DetectionLabeler.

    Attributes:
        label_source_uri: URI of the label file (e.g., COCO JSON or VOC XML list).
        input_format: Format of the input label file ("coco_json", "voc_xml", "yolo_txt").
        normalize_bbox: If True, normalize BBox to [0, 1].
    """
    label_source_uri: str = Field(..., description="URI of the label file.")
    input_format: Literal["coco_json", "voc_xml", "yolo_txt"] = Field("coco_json", description="Format of the input label file.")
    normalize_bbox: bool = Field(True, description="Normalize BBox coordinates to [0, 1].")

class OCRLabelerConfig(BaseModel):
    """
    Configuration for OCRLabeler.

    Attributes:
        label_source_uri: URI of the OCR label file (e.g., JSON containing text and bbox).
        tokenizer_config: Configuration dictionary for the Pre-trained Tokenizer.
        max_sequence_length: Maximum token sequence length for padding.
        padding_token: The token used for padding.
    """
    label_source_uri: str = Field(..., description="URI of the OCR label file.")
    tokenizer_config: Dict[str, Any] = Field({}, description="Configuration for Pre-trained Tokenizer.")
    max_sequence_length: conint(gt=0) = Field(128, description="Maximum token sequence length, used for padding.")
    padding_token: str = Field("<pad>", description="Token used for padding.")
    
class SegmentationLabelerConfig(BaseModel):
    """
    Configuration for SegmentationLabeler.

    Attributes:
        label_source_uri: URI of the label file (often RLE or raw mask paths).
        mask_encoding: Encoding format for the segmentation mask ("rle", "png_mask", "polygons").
    """
    label_source_uri: str = Field(..., description="URI of the label file.")
    mask_encoding: Literal["rle", "png_mask", "polygons"] = Field("png_mask", description="Encoding format for the segmentation mask.")

class EmbeddingLabelerConfig(BaseModel):
    """
    Configuration for EmbeddingLabeler.

    Attributes:
        label_source_uri: URI of the file containing target IDs and/or clustering info.
        vector_dim: Expected dimension of the embedding vector (must be > 1).
    """
    label_source_uri: str = Field(..., description="URI of the file containing target IDs and/or clustering info.")
    vector_dim: conint(gt=1) = Field(512, description="Expected dimension of the embedding vector.")


# --- 2. Main Config Schema: General structure for Labeling Task ---

class LabelerConfig(BaseModel):
    """
    Master schema for a single Labeler configuration in CV_Factory.
    
    Uses the 'params' field to hold task-specific configuration, enforced via 
    a root validator for semantic consistency.

    Attributes:
        task_type: The CV task type (e.g., classification, detection).
        params: Task-specific configuration (validated against Sub-Schemas).
        validation_ratio: Ratio of label samples to be randomly validated (0.0 to 1.0).
        cache_path: Optional path to cache loaded and standardized labels.
    """
    
    task_type: Literal["classification", "detection", "segmentation", "ocr", "embedding"] = Field(
        ..., description="The CV task type (e.g., classification, detection)."
    )
    
    # Union includes all specific config schemas
    params: Union[
        ClassificationLabelerConfig,
        DetectionLabelerConfig,
        OCRLabelerConfig,
        SegmentationLabelerConfig, 
        EmbeddingLabelerConfig
    ] = Field(..., description="Detailed configuration specific to the task type.")
    
    validation_ratio: confloat(ge=0, le=1) = Field(0.01, description="Ratio of label samples to be randomly checked (validation).")
    cache_path: Optional[str] = Field(None, description="Path to cache loaded labels (e.g., as Parquet).")
    
    class Config:
        """Pydantic configuration."""
        # Hardening: Forbid extra fields for strict configuration
        extra = "forbid" 
        
    @root_validator(pre=True)
    def validate_params_type_match(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule: Enforce that the 'params' dictionary matches the structure 
        required by the declared 'task_type'. (Semantic Validation)
        
        This overrides Pydantic's default Union behavior to guarantee semantic consistency.
        """
        task_type = values.get('task_type')
        params_dict = values.get('params')

        if not task_type or not params_dict:
            return values

        type_map: Dict[str, Type[BaseModel]] = {
            "classification": ClassificationLabelerConfig,
            "detection": DetectionLabelerConfig,
            "segmentation": SegmentationLabelerConfig,
            "ocr": OCRLabelerConfig,
            "embedding": EmbeddingLabelerConfig,
        }

        expected_type = type_map.get(task_type)

        if expected_type:
            try:
                # Attempt to parse the raw 'params' dict into the expected specific model
                expected_type(**params_dict)
            except ValidationError as e:
                # If parsing fails, the structure is wrong for the task type
                raise ValueError(
                    f"Configuration mismatch: task_type '{task_type}' requires '{expected_type.__name__}' "
                    f"structure in 'params', but validation failed: {e}"
                )
        
        # NOTE: Pydantic will handle the final conversion of 'params' to the Union type instance after this pre-validation.
        return values


# --- 3. Wrapper for a list of Labelers ---

class LabelingProjectConfig(BaseModel):
    """
    Overall configuration for the entire Labeling Project, supporting multiple pipelines/tasks.

    Attributes:
        labeling_pipelines: List of labeler configurations, one for each task type.
    """
    labeling_pipelines: List[LabelerConfig] = Field(
        ..., description="List of labeling pipelines, one for each task type."
    )