# cv_factory/shared_libs/ml_core/trainer/implementations/contrastive_trainer.py

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared_libs.ml_core.trainer.base.base_distributed_trainer import BaseDistributedTrainer # <<< NEW INHERITANCE >>>
from shared_libs.ml_core.trainer.utils import distributed_utils 

# Thêm imports cho Schema
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

logger = logging.getLogger(__name__)

class ContrastiveTrainer(BaseDistributedTrainer): # <<< Inherits DDP setup and wrapping >>>
    """
    A trainer for contrastive learning methods (e.g., SimCLR, BYOL).
    DDP support is handled by the parent class.
    """
    def __init__(self, 
                 model: nn.Module, 
                 trainer_config: TrainerConfig, 
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the ContrastiveTrainer.
        """
        # 1. Base Init: Handles device, optimizer, scheduler, loss, AND DDP setup/wrap.
        super().__init__(model, trainer_config, model_config, **kwargs)
        
        # 2. EXTRACT CONFIG from SCHEMA
        params = trainer_config.params if trainer_config.params else {}
        self.temperature = params.get("temperature", 0.07)
        
        if distributed_utils.is_main_process():
            logger.info(f"ContrastiveTrainer initialized with temperature={self.temperature}.")

    # Implementation of the abstract method from BaseDistributedTrainer
    def _run_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Runs the core contrastive training loop for a single epoch."""
        self.model.train()
                
        for batch in train_loader:
            # Assuming batch contains positive pairs (e.g., [img_a, img_b])
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                img_a, img_b = batch[:2]
            else:
                logger.warning("Contrastive batch format incorrect, skipping step.")
                continue
                
            img_a, img_b = img_a.to(self.device), img_b.to(self.device)
            
            # Forward pass to get embeddings
            embedding_a = self.model(img_a)
            embedding_b = self.model(img_b)
            
            # Compute contrastive loss
            loss = self.loss_fn(embedding_a, embedding_b, self.temperature)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if val_loader:
            self.evaluate(val_loader)
            
    # NOTE: train_step abstract method needs definition:
    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
         raise NotImplementedError("ContrastiveTrainer handles training via _run_epoch method.")

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
         raise NotImplementedError("ContrastiveTrainer must implement evaluate.")
             
    # --- CHECKPOINTING (DELEGATED TO PARENT) ---
    def save_checkpoint(self, path: str) -> None:
        """Saves checkpoint, executed by Rank 0 only, ensuring DDP model unwrapping."""
        if distributed_utils.is_main_process():
            super().save(path)
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads checkpoint across all ranks, ensuring proper device mapping."""
        super().load(path)
        distributed_utils.synchronize_between_processes("load_checkpoint")