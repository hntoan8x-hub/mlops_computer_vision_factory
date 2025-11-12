# cv_factory/shared_libs/ml_core/trainer/configs/trainer_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, PositiveInt, constr
from typing import List, Dict, Any, Union, Literal, Optional
import logging
import torch # Need torch for distributed validation check

# Import modules from the new configuration folder
from .optimizer_config_schema import OptimizerConfig, SchedulerConfig
from .checkpoint_config_schema import CheckpointConfig
from .distributed_config_schema import DistributedConfig
# Giả định BaseConfig đã được định nghĩa ở một nơi chung, nhưng ta định nghĩa lại để độc lập

class BaseConfig(BaseModel):
    """Base schema for general component parameters."""
    class Config:
        extra = "allow" 

class TrainerConfig(BaseConfig):
    """Master Schema for the specific Trainer implementation (e.g., CNNTrainer, SemiSupervisedTrainer)."""
    
    # 1. Trainer Identification
    type: constr(to_lower=True) = Field(..., description="Trainer implementation (e.g., 'cnn', 'finetune', 'semi_supervised').")
    
    # 2. Execution Parameters
    epochs: PositiveInt = Field(10, description="Total number of training epochs.")
    batch_size: PositiveInt = Field(32, description="The effective batch size per GPU/process.")
    
    # 3. Nested Configurations (Hardened Contracts)
    distributed: DistributedConfig = Field(..., description="Distributed training configuration.")
    checkpoint: CheckpointConfig = Field(..., description="Checkpoint saving and loading rules.")
    optimizer: OptimizerConfig = Field(..., description="Model optimizer configuration.")
    scheduler: SchedulerConfig = Field(..., description="Learning rate scheduler configuration.")
    
    # 4. Custom Parameters (For Loss Fn, Freeze Layers, SSL Weights, etc.)
    # These parameters are usually passed directly to the specific Trainer's __init__ via self.params
    
    # Custom validation rules
    
    @validator('type')
    def validate_trainer_consistency(cls, v, values):
        """Rule: Ensure special trainers have required parameters in the general params dictionary."""
        # Note: Logic kiểm tra cụ thể cần truy cập self.params của BaseConfig (nếu có)
        params = values.get('params', {})
        
        if v == "semi_supervised":
            if 'consistency_weight' not in params:
                logger.warning("Semi-supervised trainer configured without 'consistency_weight'. Defaulting to 1.0.")
        
        if v == "finetune":
             if 'num_layers_to_unfreeze' not in params:
                logger.warning("Finetune trainer configured without 'num_layers_to_unfreeze'. Defaulting to 1.")
        
        return v