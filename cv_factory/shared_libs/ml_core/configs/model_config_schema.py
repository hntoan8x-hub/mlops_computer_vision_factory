# cv_factory/shared_libs/ml_core/configs/model_config_schema.py

from pydantic import Field, validator, BaseModel, constr, NonNegativeInt
from typing import List, Dict, Any, Union, Literal, Optional

# Assuming BaseConfig is available
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


class ModelConfig(BaseConfig):
    """
    Master Schema for model architecture and loading configuration.
    This schema dictates which model (type, name) and weights (source) should be used.
    """
    
    # 1. Model Identification and Type
    type: constr(to_lower=True) = Field(..., description="Model family type (e.g., 'cnn', 'transformer', 'r-cnn').")
    name: str = Field(..., description="Specific model identifier (e.g., 'resnet50', 'google/vit-base-patch16-224').")
    
    # 2. Weights and Source Control
    weights_source: Literal["pretrained", "mlflow_registry", "local_file", "scratch"] = Field(
        "pretrained", 
        description="Source to load weights from (e.g., Hugging Face/Torchvision pretrained, MLflow, or train from scratch)."
    )
    
    weights_uri: Optional[str] = Field(None, description="URI or name/version if weights_source is 'mlflow_registry' or 'local_file'.")
    
    # 3. Task and Output
    task_type: Literal["classification", "detection", "segmentation", "embedding"] = Field(
        "classification", 
        description="The primary CV task this model is configured for."
    )
    num_classes: NonNegativeInt = Field(10, description="Number of output classes (for classification/segmentation).")
    
    # Custom Validation Rules (Rules of the Game)
    
    @validator('weights_uri')
    def validate_uri_if_needed(cls, v, values):
        """Rule: Enforce weights_uri to be present if loading from MLflow or local file."""
        source = values.get('weights_source')
        if source in ["mlflow_registry", "local_file"] and not v:
            raise ValueError(f"If weights_source is '{source}', 'weights_uri' must be provided.")
        return v

    @validator('task_type')
    def validate_task_specifics(cls, v, values):
        """Rule: Ensure num_classes is consistent with the task type."""
        num_classes = values.get('num_classes')
        if v == "embedding" and num_classes is not None and num_classes > 0:
            logger.warning(
                f"Task '{v}' typically outputs a feature vector, not classes. 'num_classes' is irrelevant."
            )
        # Add rules for detection/segmentation output checks if needed
        return v