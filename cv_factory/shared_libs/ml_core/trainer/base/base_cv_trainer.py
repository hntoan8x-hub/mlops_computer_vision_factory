# cv_factory/shared_libs/ml_core/trainer/base/base_cv_trainer.py

import abc
import logging
from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from .base_trainer import BaseTrainer
# Import Utilities from their hardened locations
from ..utils import optimizer_utils     # For creating Optimizer and Scheduler
from ..utils import checkpoint_utils    # For checkpoint I/O
from ..utils import distributed_utils   # For DDP awareness

logger = logging.getLogger(__name__)

class BaseCVTrainer(BaseTrainer, abc.ABC):
    """
    Abstract Base Class for all Computer Vision model trainers.

    Extends BaseTrainer to enforce PyTorch model handling, integrating utility-based 
    initialization for Optimizer, Scheduler, and Checkpointing, ensuring 
    all concrete trainers are DDP-ready and auditable.
    """

    def __init__(self, model: torch.nn.Module, **kwargs: Dict[str, Any]):
        """
        Initializes the trainer.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            **kwargs: Configuration dictionary, must contain 'optimizer_config' 
                      and 'loss_fn_config' for automated initialization.
        """
        # 1. Initialize BaseTrainer (usually handles generic config)
        super().__init__(**kwargs) 
        
        self.model = model
        # Determine device based on config or DDP environment local rank
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # --- 2. Initialize Optimizer and Scheduler using Utilities (CRITICAL HARDENING) ---
        optimizer_config = kwargs.get("optimizer_config", {})
        scheduler_config = kwargs.get("scheduler_config", {})

        self.optimizer: optim.Optimizer = optimizer_utils.get_optimizer(self.model, optimizer_config)
        self.scheduler = optimizer_utils.get_scheduler(self.optimizer, scheduler_config)
        
        # 3. Initialize Loss Function
        loss_fn_config = kwargs.get("loss_fn_config", {"type": "CrossEntropyLoss"})
        self.loss_fn: nn.Module = self._get_loss_fn(loss_fn_config)

        logger.info(f"Trainer initialized on device: {self.device}. Optimizer: {type(self.optimizer).__name__}")

    def _get_loss_fn(self, config: Dict[str, Any]) -> nn.Module:
        """Helper to create loss function from configuration."""
        loss_type = config.get('type', 'CrossEntropyLoss').lower()
        
        # Add production-relevant loss functions here
        if loss_type == 'crossentropyloss':
            return nn.CrossEntropyLoss(weight=config.get('weight'))
        elif loss_type == 'mseloss':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_type}")

    @abc.abstractmethod
    def fit(self, train_loader: torch.utils.data.DataLoader, 
            val_loader: Optional[torch.utils.data.DataLoader] = None, 
            epochs: int = 10) -> None:
        """
        [ABSTRACT] Trains the model. This method must be implemented by subclasses 
        to define the specific training loop (e.g., standard CNN loop, transformer loop).
        """
        raise NotImplementedError

    # train_step and evaluate implementations (if provided) rely on self.optimizer/self.loss_fn

    # --- Checkpoint Management (Using Utilities) ---

    def save(self, path: str, **kwargs: Dict[str, Any]) -> None:
        """Saves the model, optimizer, and scheduler state using checkpoint utilities."""
        
        # CRITICAL: Unwrap DDP model if necessary before saving the state dict
        model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        
        state = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'device': self.device.type,
            **kwargs
        }
        # Delegation to utility for safe I/O
        checkpoint_utils.save_checkpoint(state, path)
        
        # DDP Synchronization: In concrete trainers, a distributed_utils.synchronize_between_processes
        # call should follow this save if it's executed on Rank 0.

    def load(self, path: str, **kwargs: Dict[str, Any]) -> None:
        """Loads the model, optimizer, and scheduler state using checkpoint utilities."""
        
        # Delegation to utility for safe I/O
        state = checkpoint_utils.load_checkpoint(path)
        
        # Load weights onto the current device (required for DDP compliance)
        map_location = self.device if self.device.type == 'cuda' else 'cpu'
        
        # Determine which model object to load the state into (raw model or model.module)
        model_to_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        
        model_to_load.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if self.scheduler and state.get('scheduler_state_dict'):
             self.scheduler.load_state_dict(state['scheduler_state_dict'])
             
        logging.info(f"Checkpoint loaded to device {self.device}.")
        
        # DDP Synchronization: In concrete trainers, a distributed_utils.synchronize_between_processes
        # call should follow this load.