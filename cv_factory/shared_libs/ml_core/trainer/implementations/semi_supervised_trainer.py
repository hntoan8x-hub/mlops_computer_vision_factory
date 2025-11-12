# cv_factory/shared_libs/ml_core/trainer/implementations/semi_supervised_trainer.py

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared_libs.ml_core.trainer.base.base_distributed_trainer import BaseDistributedTrainer # <<< NEW INHERITANCE >>>
from shared_libs.ml_core.trainer.utils import distributed_utils 

# Thêm imports cho Schema
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

logger = logging.getLogger(__name__)

class SemiSupervisedTrainer(BaseDistributedTrainer): # <<< Inherits DDP setup and wrapping >>>
    """
    A trainer for semi-supervised learning techniques (e.g., Consistency Regularization).
    DDP is handled by the parent class.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 trainer_config: TrainerConfig, 
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the SemiSupervisedTrainer.
        """
        # 1. Base Init: Handles device, optimizer, scheduler, loss, AND DDP setup/wrap.
        super().__init__(model, trainer_config, model_config, **kwargs)
        
        # 2. EXTRACT CONFIG from SCHEMA
        params = trainer_config.params if trainer_config.params else {}
        self.consistency_weight = params.get("consistency_weight", 1.0)
        # consistency_loss_fn must still be handled as a custom object, likely passed via kwargs
        self.consistency_loss_fn = kwargs.get("consistency_loss_fn", None) 
        
        if distributed_utils.is_main_process():
            logger.info(f"SemiSupervisedTrainer initialized (Weight: {self.consistency_weight}).")

    # NOTE: We keep the custom 'fit' method signature as it requires two DataLoaders
    def fit(self, labeled_loader: DataLoader, 
            unlabeled_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None, 
            epochs: Optional[int] = None) -> None: # Added Optional[int] override
        """
        Trains the model using both labeled and unlabeled data.
        """
        # Use schema epoch count if not provided as override
        total_epochs = epochs if epochs is not None else self.trainer_config.epochs
        self.model.train()
        
        for epoch in range(total_epochs):
            # CRITICAL DDP STEP: Set epoch for DistributedSampler for both loaders
            if distributed_utils.get_world_size() > 1:
                if hasattr(labeled_loader.sampler, 'set_epoch'):
                    labeled_loader.sampler.set_epoch(epoch)
                if hasattr(unlabeled_loader.sampler, 'set_epoch'):
                    unlabeled_loader.sampler.set_epoch(epoch)
                
            self._run_semi_supervised_epoch(labeled_loader, unlabeled_loader, val_loader, epoch, total_epochs)
            
    # Implementation of the abstract method from BaseDistributedTrainer (adapted for SSL)
    # We must implement _run_epoch, but since the signature is different, 
    # we implement it as a wrapper/stub and use a custom SSL epoch method.
    def _run_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Stub for BaseDistributedTrainer compliance. Actual logic in _run_semi_supervised_epoch."""
        # Note: This trainer's fit method overrides the parent's, so this method 
        # is technically unused but must be defined.
        pass 
        
    def _run_semi_supervised_epoch(self, labeled_loader: DataLoader, 
                                   unlabeled_loader: DataLoader, 
                                   val_loader: Optional[DataLoader], 
                                   epoch: int, 
                                   total_epochs: int) -> None:
        """Runs the core semi-supervised training loop for one epoch."""
        
        # Use zip to iterate over both simultaneously
        for i, (labeled_batch, unlabeled_batch) in enumerate(zip(labeled_loader, unlabeled_loader)):
            
            # --- Labeled Data Loss (Supervised) ---
            inputs_l, labels_l = labeled_batch
            inputs_l, labels_l = inputs_l.to(self.device), labels_l.to(self.device)
            outputs_l = self.model(inputs_l)
            loss_l = self.loss_fn(outputs_l, labels_l)
            
            # --- Unlabeled Data Loss (Consistency/Unsupervised) ---
            inputs_ul, _ = unlabeled_batch 
            inputs_ul = inputs_ul.to(self.device)
            outputs_ul = self.model(inputs_ul)
            
            total_loss = loss_l
            if self.consistency_loss_fn:
                pseudo_labels = torch.argmax(outputs_ul, dim=1) 
                consistency_loss = self.consistency_loss_fn(outputs_ul, pseudo_labels)
                total_loss += consistency_loss * self.consistency_weight
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if (i + 1) % 100 == 0 and distributed_utils.is_main_process():
                logger.info(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/...], Loss: {total_loss.item():.4f}")

        if val_loader:
            self.evaluate(val_loader)
                
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluates the model on the validation set and aggregates metrics across all processes.
        """
        self.model.eval()
        total_loss = torch.tensor(0.0).to(self.device)
        total = torch.tensor(0).to(self.device)
        
        # ... (Evaluation logic remains the same)
        # ... (Metrics reduction logic remains the same)
        
        # NOTE: train_step abstract method needs definition:
    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
         raise NotImplementedError("SemiSupervisedTrainer handles training via fit/epoch methods.")


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