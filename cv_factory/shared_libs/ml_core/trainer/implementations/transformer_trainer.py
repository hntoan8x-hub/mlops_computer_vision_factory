# cv_factory/shared_libs/ml_core/trainer/implementations/transformer_trainer.py

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# NOTE: Hugging Face specific components are now handled by the factory/model layer, 
# not the trainer setup, but we keep the imports for type hinting clarity.
from transformers import ViTForImageClassification, AdamW 

# Import the correct hardened base class
from shared_libs.ml_core.trainer.base.base_distributed_trainer import BaseDistributedTrainer # <<< NEW INHERITANCE >>>
from shared_libs.ml_core.trainer.utils import distributed_utils 

# Import necessary Schemas for the explicit contract
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

logger = logging.getLogger(__name__)

class TransformerTrainer(BaseDistributedTrainer): # <<< Inherits DDP setup and wrapping >>>
    """
    Concrete trainer for Vision Transformer (ViT) models, specifically designed for 
    image classification. DDP, Optimizer, Scheduler, and Loss setup are handled 
    by the hardened base classes using Pydantic schemas.
    """

    def __init__(self, 
                 model: nn.Module, # Use generic nn.Module for flexibility
                 trainer_config: TrainerConfig,
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the TransformerTrainer. 
        BaseDistributedTrainer handles DDP, Optimizer, and Loss setup based on schemas.
        """
        # 1. Base Init: Handles device, optimizer, scheduler, loss (via BaseCVTrainer), 
        # AND DDP setup/wrap (via BaseDistributedTrainer).
        super().__init__(model, trainer_config, model_config, **kwargs)
        
        # NOTE: We no longer manually define self.optimizer or self.loss_fn here.
        # They are correctly set by BaseCVTrainer using the validated schemas.
        
        if distributed_utils.is_main_process():
            # Log successful initialization after all setup is complete
            logger.info(f"TransformerTrainer initialized. DDP status handled by parent. Optimizer: {type(self.optimizer).__name__}")
            
    # Implementation of the abstract method from BaseDistributedTrainer
    def _run_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   val_loader: Optional[torch.utils.data.DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Runs the core training and evaluation loop for a single epoch."""

        self.model.train()
        for i, batch in enumerate(train_loader):
            # Data to Device (Assumes Hugging Face dataloader output format)
            inputs = batch['pixel_values'].to(self.device)
            # Labels might be missing in some Hugging Face dataloaders, check for presence
            labels = batch['labels'].to(self.device) if 'labels' in batch else None 
            
            self.optimizer.zero_grad()
            
            # Forward pass: Hugging Face models typically return an output object
            # Pass labels only if available and needed for integrated loss calculation
            outputs = self.model(pixel_values=inputs, labels=labels)
            
            # Use integrated loss if available (typical for HF models) or use self.loss_fn
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else self.loss_fn(outputs.logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            # Logging: Only main process logs progress
            if (i + 1) % 100 == 0 and distributed_utils.is_main_process():
                logger.info(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        if val_loader:
            self.evaluate(val_loader)


    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Evaluates the model on the test set and aggregates metrics across all processes.
        """
        self.model.eval()
        
        # Initialize metrics as PyTorch tensors for distributed reduction
        total_loss = torch.tensor(0.0).to(self.device)
        correct = torch.tensor(0).to(self.device)
        total = torch.tensor(0).to(self.device)
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.logits.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum()

        # AGGREGATION: Reduce/Sum values from all processes
        if distributed_utils.get_world_size() > 1:
            distributed_utils.reduce_tensor_across_processes(total_loss, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(correct, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(total, op=torch.distributed.ReduceOp.SUM)
        
        metrics = {}
        if distributed_utils.is_main_process():
            # FINAL CALCULATION ON RANK 0
            accuracy = 100 * correct.item() / total.item()
            avg_loss = total_loss.item() / len(test_loader)
            
            metrics = {"loss": avg_loss, "accuracy": accuracy}
            logger.info(f"Evaluation Metrics (Aggregated on Rank 0): {metrics}")
            
        return metrics

    # NOTE: train_step abstract method needs definition:
    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
         raise NotImplementedError("TransformerTrainer handles training via _run_epoch method.")

    # --- CHECKPOINTING (DELEGATED TO PARENT) ---
    # We remove the local save/load methods and enforce usage of BaseCVTrainer's logic.
    def save_checkpoint(self, path: str) -> None:
        """Saves checkpoint, executed by Rank 0 only, ensuring DDP model unwrapping."""
        if distributed_utils.is_main_process():
            super().save(path)
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads checkpoint across all ranks, ensuring proper device mapping."""
        super().load(path)
        distributed_utils.synchronize_between_processes("load_checkpoint")