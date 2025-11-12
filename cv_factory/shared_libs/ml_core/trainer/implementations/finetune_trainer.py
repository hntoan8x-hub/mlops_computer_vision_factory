# cv_factory/shared_libs/ml_core/trainer/implementations/finetune_trainer.py

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from shared_libs.ml_core.trainer.base.base_distributed_trainer import BaseDistributedTrainer # <<< NEW INHERITANCE >>>
from shared_libs.ml_core.trainer.utils import distributed_utils 

# Thêm imports cho Schema
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

logger = logging.getLogger(__name__)

class FinetuneTrainer(BaseDistributedTrainer): # <<< Inherits DDP setup and wrapping >>>
    """
    Trainer for fine-tuning pre-trained models. DDP is handled by the parent class.
    Handles freezing/unfreezing layers based on schema parameters.
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 trainer_config: TrainerConfig, 
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the FinetuneTrainer.
        """
        # 1. Base Init: Handles device, optimizer, scheduler, loss, AND DDP setup/wrap.
        super().__init__(model, trainer_config, model_config, **kwargs)

        # 2. EXTRACT CONFIG from SCHEMA
        params = trainer_config.params if trainer_config.params else {}
        # Ensure we read the parameter validated by the TrainerConfig schema
        self.num_layers_to_unfreeze = params.get("num_layers_to_unfreeze", 1) 
        
        # 3. FREEZE LAYERS (Must happen before any training loop starts)
        # Note: DDP wrapping is already done by the parent __init__, but freezing should
        # be performed on the raw model if possible. Since BaseDistributedTrainer wraps 
        # *after* BaseCVTrainer init, this is safe to run here.
        self._freeze_layers()
        
        if distributed_utils.is_main_process():
            logger.info(f"Finetune Trainer initialized. Freezing completed ({self.num_layers_to_unfreeze} layers unfrozen).")

    def _freeze_layers(self) -> None:
        """
        Freezes all layers except the last N layers or the classification head.
        This operation is performed on the internal model (model.module if DDP wrapped).
        """
        model_to_freeze = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        # Freeze all parameters initially
        for param in model_to_freeze.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N layers
        unfrozen_layers = list(model_to_freeze.children())[-self.num_layers_to_unfreeze:]
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Finetune trainer prepared: Freezing all but the last {self.num_layers_to_unfreeze} layers.")

    # Implementation of the abstract method from BaseDistributedTrainer
    def _run_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Runs the core fine-tuning loop for a single epoch."""
        self.model.train()
        
        # Placeholder for actual training loop logic (e.g., calling self.train_step)
        if distributed_utils.is_main_process():
            logger.info(f"Epoch {epoch+1}/{total_epochs} starting fine-tuning loop...")

        # NOTE: train_step/evaluate implementations are assumed to exist or be implemented via mixins.
        # For this refactor, we ensure the DDP boilerplate is handled by the parent 'fit'.
        
        if val_loader:
             self.evaluate(val_loader)
             
    # --- CHECKPOINTING (DELEGATED TO PARENT) ---
    # We remove the local save/load methods to enforce usage of BaseCVTrainer's logic.
    def save_checkpoint(self, path: str) -> None:
        """Saves checkpoint, executed by Rank 0 only, ensuring DDP model unwrapping."""
        if distributed_utils.is_main_process():
            super().save(path)
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads checkpoint across all ranks, ensuring proper device mapping."""
        super().load(path)
        distributed_utils.synchronize_between_processes("load_checkpoint")

    # NOTE: train_step and evaluate need concrete implementation if not provided by mixins.
    # We assume they exist for now, following the original intent of the base classes.
    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
         raise NotImplementedError("FinetuneTrainer must implement train_step.")

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
         raise NotImplementedError("FinetuneTrainer must implement evaluate.")