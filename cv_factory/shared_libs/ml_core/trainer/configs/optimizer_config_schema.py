# cv_factory/shared_libs/ml_core/trainer/configs/optimizer_config_schema.py

from pydantic import Field, BaseModel, validator, constr
from typing import Dict, Any, Union, Literal, Optional

class OptimizerConfig(BaseModel):
    """Schema for the model optimizer configuration."""
    class Config:
        extra = "forbid"

    name: Literal["Adam", "AdamW", "SGD", "RMSprop"] = Field("AdamW", description="Optimizer type.")
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-1, description="Initial learning rate.")
    weight_decay: float = Field(default=0.01, ge=0.0, description="L2 regularization/Weight decay.")
    
    # SGD specific parameters
    momentum: float = Field(default=0.9, ge=0.0, le=1.0, description="Momentum factor (for SGD).")
    nesterov: bool = Field(False, description="Enables Nesterov momentum (for SGD).")

class SchedulerConfig(BaseModel):
    """Schema for the learning rate scheduler configuration."""
    class Config:
        extra = "forbid"

    name: Optional[Literal["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]] = Field(
        None, description="Scheduler type, or None if no scheduler is used."
    )
    
    # StepLR parameters
    step_size: PositiveInt = Field(10, description="Period of learning rate decay (for StepLR).")
    gamma: float = Field(0.1, ge=0.0, le=1.0, description="Multiplicative factor of learning rate decay (for StepLR).")

    # ReduceLROnPlateau parameters
    mode: Literal["min", "max"] = Field("min", description="Mode for ReduceLROnPlateau.")
    patience: NonNegativeInt = Field(10, description="Number of epochs with no improvement after which learning rate will be reduced.")
    factor: float = Field(0.1, ge=0.0, le=1.0, description="Factor by which the learning rate will be reduced.")