# cv_factory/shared_libs/ml_core/configs/trainer_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, PositiveInt, constr
from typing import List, Dict, Any, Union, Literal, Optional
import logging

logger = logging.getLogger(__name__)

# Assuming BaseConfig is available and used for overall structure
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


# --- 1. Distributed and Device Rules ---

class DistributedConfig(BaseConfig):
    """Rules for Distributed Training setup (DDP)."""
    type: constr(to_lower=True) = Field("ddp", const=True, description="Distributed framework type (currently only DDP).")
    backend: Literal["nccl", "gloo"] = Field("nccl", description="Backend for distributed communication (nccl for GPU, gloo for CPU).")
    world_size: PositiveInt = Field(1, description="Total number of processes/GPUs to use for training.")
    
    @validator('backend')
    def validate_nccl_if_gpu(cls, v, values):
        # Rule: nccl is mandatory for performance if GPUs are used (implied by world_size > 1)
        if v == "gloo" and values.get('world_size', 1) > 1 and torch.cuda.is_available():
            logger.warning("Using 'gloo' backend in distributed mode with available GPUs might lead to suboptimal performance. 'nccl' is recommended.")
        return v

# --- 2. Hyperparameter and Optimizer Rules ---

class OptimizerConfig(BaseModel):
    """Rules for the model optimizer."""
    type: Literal["Adam", "AdamW", "SGD"] = Field("AdamW", description="Optimizer type.")
    learning_rate: Field(default=1e-4, ge=1e-6, le=1e-1) = Field(description="Initial learning rate.")
    weight_decay: Field(default=0.01, ge=0.0) = Field(description="L2 regularization/Weight decay.")
    
class CheckpointConfig(BaseModel):
    """Rules for checkpointing the model."""
    save_path: str = Field(..., description="Base path/URI to save checkpoints.")
    frequency_epochs: NonNegativeInt = Field(1, description="Save checkpoint every N epochs.")
    max_checkpoints: NonNegativeInt = Field(3, description="Maximum number of checkpoints to retain.")


# --- 3. Master Trainer Configuration ---

class TrainerConfig(BaseConfig):
    """Master Schema for the specific Trainer implementation."""
    
    # The concrete DDP-ready trainer class name
    type: constr(to_lower=True) = Field(..., description="Trainer implementation (e.g., 'cnn_trainer', 'transformer_trainer').")
    
    # Checkpoint and Distributed configurations are now nested
    distributed: DistributedConfig = Field(..., description="Distributed training configuration.")
    checkpoint: CheckpointConfig = Field(..., description="Checkpoint saving and loading rules.")
    optimizer: OptimizerConfig = Field(..., description="Model optimizer configuration.")
    
    # Generic execution parameters (often passed via BaseConfig.params or here)
    epochs: PositiveInt = Field(10, description="Total number of training epochs.")
    batch_size: PositiveInt = Field(32, description="The effective batch size per GPU/process.")
    
    # Custom validator to check consistency between trainer type and required parameters
    @validator('type')
    def validate_trainer_consistency(cls, v, values):
        # Rule: Ensure special trainers have required parameters
        if v == "semi_supervised_trainer":
            if 'consistency_weight' not in values.get('params', {}):
                raise ValueError("Semi-supervised trainer requires 'consistency_weight' parameter in params.")
        # Add more type-specific checks here
        return v