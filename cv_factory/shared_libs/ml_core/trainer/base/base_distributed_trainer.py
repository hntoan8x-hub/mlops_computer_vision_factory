# cv_factory/shared_libs/ml_core/trainer/base/base_distributed_trainer.py

import abc
import logging
from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Import validated configuration schemas
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     
# Import necessary utils
from ..utils import distributed_utils
from .base_cv_trainer import BaseCVTrainer

logger = logging.getLogger(__name__)

class BaseDistributedTrainer(BaseCVTrainer, abc.ABC):
    """
    Abstract Base Class for distributed model trainers.

    Extends BaseCVTrainer by enforcing DDP setup and model wrapping 
    based on the validated configuration. Subclasses must override 
    train_step/evaluate but no longer handle DDP boilerplate.
    """

    def __init__(self, 
                 model: nn.Module, 
                 trainer_config: TrainerConfig, 
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the distributed trainer, sets up the environment, and wraps the model.
        
        Args:
            model (nn.Module): The PyTorch model.
            trainer_config (TrainerConfig): Validated Pydantic object for trainer settings.
            model_config (ModelConfig): Validated Pydantic object for model architecture settings.
            **kwargs: Configuration arguments, including runtime 'rank' if provided.
        """
        # CRITICAL: Call super init first (handles device assignment, optimizer, scheduler, loss)
        super().__init__(model=model, trainer_config=trainer_config, model_config=model_config, **kwargs)
        
        # 1. SETUP DISTRIBUTED ENVIRONMENT
        # Distributed setup uses environment variables and the backend from the config schema.
        self.world_size = trainer_config.distributed.world_size
        self.rank = kwargs.get("rank", 0) # Rank is typically runtime information
        
        if self.world_size > 1:
            # We use the utility to initialize the process group
            distributed_utils.setup_distributed_environment(
                backend=trainer_config.distributed.backend
            )
            
            # 2. WRAP MODEL FOR DDP
            # The model is already on the correct device (self.device from BaseCVTrainer).
            # We use the utility's logic to wrap the model and move it to the correct local rank device.
            # NOTE: We skip self.model = distributed_utils.wrap_model_for_distributed(self.model)
            # here and do the minimal wrap to respect device assignment done in BaseCVTrainer.
            
            self.model.to(self.device) # Ensure model is on the assigned device
            
            # DDP wrap must occur after process group initialization
            self.model = DDP(
                self.model, 
                device_ids=[self.device] if self.device.type == 'cuda' else None
            )
            
            logger.info(f"Model wrapped for DDP on rank {distributed_utils.get_rank()}/{distributed_utils.get_world_size()}.")

        elif distributed_utils.is_main_process():
            logger.info("Distributed setup skipped. World size is 1.")


    def fit(self, train_loader: torch.utils.data.DataLoader, 
            val_loader: Optional[torch.utils.data.DataLoader] = None, 
            epochs: Optional[int] = None) -> None:
        """
        Trains the model in a distributed manner, ensuring DistributedSampler epoch setting.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            epochs (Optional[int]): Number of training epochs (defaults to config value).
        """
        total_epochs = epochs if epochs is not None else self.trainer_config.epochs
        
        for epoch in range(total_epochs):
            # CRITICAL DDP STEP: Ensure DistributedSampler shuffles correctly
            if distributed_utils.get_world_size() > 1 and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            # Subclasses must implement the iteration loop over train_loader
            # For simplicity, we use the BaseCVTrainer's abstract signature here, 
            # and rely on concrete trainers to implement the loop structure.
            self._run_epoch(train_loader, val_loader, epoch, total_epochs)
            
    # We add a protected method to be implemented by subclasses for epoch iteration
    @abc.abstractmethod
    def _run_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   val_loader: Optional[torch.utils.data.DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Runs a single epoch of training and evaluation."""
        raise NotImplementedError