# shared_libs/ml_core/cv_model/configs/model_config_schema.py

from pydantic import Field, constr
from typing import Dict, Any, Optional
from shared_libs.ml_core.configs.base_config_schema import BaseConfig # Giả định BaseConfig tồn tại

class ModelConfig(BaseConfig):
    """
    Schema for validating Model Architecture Configuration.
    """
    name: constr(to_lower=True) = Field(..., description="The name of the model architecture (e.g., 'resnet', 'unet').")
    task_type: constr(to_lower=True) = Field(..., description="The type of task the model is designed for (e.g., 'classification', 'depth_estimation').")
    pretrained: bool = Field(False, description="Whether to load pre-trained weights.")
    num_classes: Optional[int] = Field(None, description="Number of output classes (if classification/segmentation).")
    
    # params field is used for architecture-specific settings (e.g., backbone_layers, input_channels)
    params: Optional[Dict[str, Any]] = Field(None, description="Architecture-specific parameters.")