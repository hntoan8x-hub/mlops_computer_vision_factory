import abc
import logging
from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_cv_trainer import BaseCVTrainer

logger = logging.getLogger(__name__)

class BaseDistributedTrainer(BaseCVTrainer, abc.ABC):
    """
    Abstract Base Class for distributed model trainers.

    Extends BaseCVTrainer to handle multi-GPU and multi-node training scenarios.
    """

    def __init__(self, model: nn.Module, **kwargs: Dict[str, Any]):
        super().__init__(model, **kwargs)
        self.rank = kwargs.get("rank", 0)
        self.world_size = kwargs.get("world_size", 1)
        self._setup_distributed_environment()
        self._wrap_model_for_distributed()
        
    @abc.abstractmethod
    def _setup_distributed_environment(self) -> None:
        """
        Sets up the distributed training environment (e.g., DDP, Horovod).
        """
        raise NotImplementedError

    def _wrap_model_for_distributed(self) -> None:
        """
        Wraps the model for distributed training using PyTorch's DDP.
        """
        self.model = DDP(self.model, device_ids=[self.device] if self.device.type == 'cuda' else None)
        logger.info(f"Model wrapped for distributed training on rank {self.rank}.")

    def fit(self, train_loader: torch.utils.data.DataLoader, 
            val_loader: Optional[torch.utils.data.DataLoader] = None, 
            epochs: int = 10) -> None:
        """
        Trains the model in a distributed manner.
        """
        for epoch in range(epochs):
            if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            for inputs, labels in train_loader:
                self.train_step(inputs, labels)
            
            if val_loader:
                self.evaluate(val_loader)