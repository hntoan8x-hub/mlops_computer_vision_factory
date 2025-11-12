# cv_factory/shared_libs/ml_core/trainer/configs/checkpoint_config_schema.py

from pydantic import Field, BaseModel, NonNegativeInt, constr
from typing import Optional, Literal

class CheckpointConfig(BaseModel):
    """Schema for checkpoint saving and loading rules."""
    class Config:
        extra = "forbid"
        
    save_path: str = Field(..., description="Base path/URI to save model checkpoints.")
    frequency_epochs: NonNegativeInt = Field(1, description="Save checkpoint every N epochs.")
    max_checkpoints: NonNegativeInt = Field(3, description="Maximum number of checkpoints to retain (for rotation).")
    resume_from: Optional[str] = Field(None, description="Path to a checkpoint file to resume training from.")