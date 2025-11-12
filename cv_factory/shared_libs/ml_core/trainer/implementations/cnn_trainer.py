# cv_factory/shared_libs/ml_core/trainer/implementations/cnn_trainer.py

import logging
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from shared_libs.ml_core.trainer.base.base_distributed_trainer import BaseDistributedTrainer # <<< NEW INHERITANCE >>>
from shared_libs.ml_core.trainer.utils import distributed_utils 

# CRITICAL NEW IMPORTS: Schema validation objects 
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

logger = logging.getLogger(__name__)

class CNNTrainer(BaseDistributedTrainer): # <<< Inherits DDP setup and wrapping >>>
    """
    Concrete trainer for Convolutional Neural Networks (CNNs). 
    DDP support is now handled by BaseDistributedTrainer.
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 trainer_config: TrainerConfig, 
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the CNN Trainer. BaseDistributedTrainer handles DDP setup.
        """
        # 1. Base Init: Handles device, optimizer, scheduler, loss, AND DDP setup/wrap.
        super().__init__(model, trainer_config, model_config, **kwargs)
        
        if distributed_utils.is_main_process():
            logger.info(f"CNNTrainer initialized (DDP status handled by parent). Rank {distributed_utils.get_rank()}.")

    # Implementation of the abstract method from BaseDistributedTrainer
    def _run_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Runs the core training and evaluation loop for a single epoch."""
        
        self.model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Logging: Only main process logs progress
            if (i + 1) % 100 == 0 and distributed_utils.is_main_process():
                logger.info(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        if val_loader:
            self.evaluate(val_loader)
                
    # evaluate method remains the same (handles distributed reduction correctly)
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluates the model on the test set and aggregates metrics across all processes.
        ... (Implementation follows DDP reduction pattern)
        """
        self.model.eval()
        
        total_loss = torch.tensor(0.0).to(self.device)
        correct = torch.tensor(0).to(self.device)
        total = torch.tensor(0).to(self.device)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        
        if distributed_utils.get_world_size() > 1:
            distributed_utils.reduce_tensor_across_processes(total_loss, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(correct, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(total, op=torch.distributed.ReduceOp.SUM)

        metrics = {}
        if distributed_utils.is_main_process():
            accuracy = 100 * correct.item() / total.item()
            avg_loss = total_loss.item() / len(test_loader)
            
            metrics = {"loss": avg_loss, "accuracy": accuracy}
            logger.info(f"Evaluation Metrics (Aggregated on Rank 0): {metrics}")
            
        return metrics

    # --- CHECKPOINTING (DELEGATED TO PARENT) ---
    # We remove the local save/load methods to enforce usage of BaseCVTrainer's logic.
    def save_checkpoint(self, path: str) -> None:
        """Saves the model and optimizer state. Only executed by the main process (Rank 0)."""
        if distributed_utils.is_main_process():
            super().save(path)
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads the model and optimizer state on all ranks."""
        super().load(path)
        distributed_utils.synchronize_between_processes("load_checkpoint")