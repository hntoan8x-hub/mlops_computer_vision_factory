# cv_factory/shared_libs/ml_core/trainer/implementations/finetune_trainer.py

import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from typing import Dict, Any, Optional

from shared_libs.ml_core.trainer.base.base_cv_trainer import BaseCVTrainer
from shared_libs.ml_core.trainer.utils import distributed_utils # DDP Utilities

logger = logging.getLogger(__name__)

class FinetuneTrainer(BaseCVTrainer):
    """
    Trainer for fine-tuning pre-trained models, integrated with DDP.
    Handles freezing/unfreezing layers before distributed setup.
    """

    def __init__(self, model: torch.nn.Module, **kwargs: Dict[str, Any]):
        """
        Initializes the FinetuneTrainer.

        Args:
            model (torch.nn.Module): The pre-trained model instance.
            **kwargs: Configuration arguments, including 'num_layers_to_unfreeze'.
        """
        super().__init__(model, **kwargs)
        self.num_layers_to_unfreeze = kwargs.get("num_layers_to_unfreeze", 1)
        
        # 1. FREEZE LAYERS (MUST happen BEFORE DDP wrap to manage requires_grad flags correctly)
        self._freeze_layers()

        # 2. SETUP DISTRIBUTED ENVIRONMENT
        distributed_utils.setup_distributed_environment() 
        
        # 3. WRAP MODEL FOR DDP
        # This moves the model (with updated requires_grad flags) to the correct device and wraps it.
        self.model = distributed_utils.wrap_model_for_distributed(self.model)
        
        # The optimizer is created in BaseCVTrainer, using the now DDP-wrapped model's parameters.
        if distributed_utils.is_main_process():
            logger.info(f"Finetune Trainer initialized. Freezing completed and DDP is active.")

    def _freeze_layers(self) -> None:
        """
        Freezes all layers except the last N layers or the classification head.
        This operation is performed on the raw model BEFORE DDP wrapping.
        """
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N layers
        unfrozen_layers = list(self.model.children())[-self.num_layers_to_unfreeze:]
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Finetune trainer prepared: Freezing all but the last {self.num_layers_to_unfreeze} layers.")
    
    # --- Checkpoint methods use the standard DDP pattern ---

    def save_checkpoint(self, path: str) -> None:
        """Saves checkpoint, executed by Rank 0 only, ensuring DDP model unwrapping."""
        if distributed_utils.is_main_process():
            model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            state = {'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
            torch.save(state, path)
            logger.info(f"Finetune model checkpoint saved by Rank 0 to {path}")
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads checkpoint across all ranks, ensuring proper device mapping."""
        map_location = {'cuda:%d' % 0: 'cuda:%d' % distributed_utils.get_rank()} 
        state = torch.load(path, map_location=map_location)
        model_to_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model_to_load.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded by Rank {distributed_utils.get_rank()}.")
        distributed_utils.synchronize_between_processes("load_checkpoint")