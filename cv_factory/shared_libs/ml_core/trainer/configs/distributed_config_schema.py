# cv_factory/shared_libs/ml_core/trainer/configs/distributed_config_schema.py

from pydantic import Field, BaseModel, PositiveInt, constr
from typing import Literal
import logging
import torch

logger = logging.getLogger(__name__)

class DistributedConfig(BaseModel):
    """Schema for Distributed Training setup (DDP/Multi-GPU)."""
    class Config:
        extra = "forbid"
        
    type: constr(to_lower=True) = Field("ddp", const=True, description="Distributed framework type (currently only DDP).")
    backend: Literal["nccl", "gloo"] = Field("nccl", description="Backend for distributed communication (nccl for GPU, gloo for CPU).")
    world_size: PositiveInt = Field(1, description="Total number of processes/GPUs to use for training.")
    
    @validator('backend')
    def validate_nccl_if_gpu(cls, v, values):
        """Rule: Warn if 'gloo' is used in distributed mode when GPUs are available."""
        if v == "gloo" and values.get('world_size', 1) > 1 and torch.cuda.is_available():
            logger.warning("Using 'gloo' backend in distributed mode with available GPUs might lead to suboptimal performance. 'nccl' is recommended.")
        return v