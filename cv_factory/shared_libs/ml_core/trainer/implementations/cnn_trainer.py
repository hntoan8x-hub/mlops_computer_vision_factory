# cv_factory/shared_libs/ml_core/trainer/implementations/cnn_trainer.py

import logging
import torch
import torch.nn.parallel
from typing import Dict, Any, Optional

from shared_libs.ml_core.trainer.base.base_cv_trainer import BaseCVTrainer
from shared_libs.ml_core.trainer.utils import distributed_utils # DDP Utilities

logger = logging.getLogger(__name__)

class CNNTrainer(BaseCVTrainer):
    """
    Concrete trainer for Convolutional Neural Networks (CNNs), integrated with 
    Distributed Data Parallel (DDP) support for multi-GPU training.
    """

    def __init__(self, model: torch.nn.Module, **kwargs: Dict[str, Any]):
        """
        Initializes the CNN Trainer, sets up the distributed environment, 
        and wraps the model for DDP.

        Args:
            model (torch.nn.Module): The CNN model to be trained.
            **kwargs: Configuration arguments, including optimizer settings.
        """
        super().__init__(model, **kwargs)
        
        # 1. SETUP DISTRIBUTED ENVIRONMENT
        distributed_utils.setup_distributed_environment() 
        
        # 2. WRAP MODEL FOR DDP
        # This function moves the model to the correct local GPU and wraps it.
        self.model = distributed_utils.wrap_model_for_distributed(self.model)
        
        # The optimizer must be initialized AFTER the model is moved/wrapped, 
        # using self.model.parameters()
        # (Assuming optimizer setup is handled in BaseCVTrainer or specific kwargs)
        
        if distributed_utils.is_main_process():
            logger.info(f"CNNTrainer initialized on Rank {distributed_utils.get_rank()}, World Size {distributed_utils.get_world_size()}")

    def fit(self, train_loader: torch.utils.data.DataLoader, 
            val_loader: Optional[torch.utils.data.DataLoader] = None, 
            epochs: int = 10) -> None:
        """
        Executes the main training loop.

        Args:
            train_loader (DataLoader): DataLoader for the training data (must use DistributedSampler).
            val_loader (Optional[DataLoader]): DataLoader for validation.
            epochs (int): Number of training epochs.
        """
        # Set epoch for DistributedSampler to ensure correct data shuffling for each epoch
        if distributed_utils.get_world_size() > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epochs)
            
        self.model.train()
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Logging: Only main process logs progress
                if (i + 1) % 100 == 0 and distributed_utils.is_main_process():
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            if val_loader:
                self.evaluate(val_loader)
                
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Evaluates the model on the test set and aggregates metrics across all processes.

        Args:
            test_loader (DataLoader): DataLoader for the test/validation data.
            
        Returns:
            Dict[str, Any]: Aggregated evaluation metrics (only returned by the main process).
        """
        self.model.eval()
        
        # Initialize metrics as PyTorch tensors for distributed reduction
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
                correct += (predicted == labels).sum() # Accumulate tensor sum
        
        # 1. AGGREGATION: Reduce/Sum values from all processes
        if distributed_utils.get_world_size() > 1:
            distributed_utils.reduce_tensor_across_processes(total_loss, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(correct, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(total, op=torch.distributed.ReduceOp.SUM)

        metrics = {}
        if distributed_utils.is_main_process():
            # 2. FINAL CALCULATION ON RANK 0
            accuracy = 100 * correct.item() / total.item()
            avg_loss = total_loss.item() / len(test_loader)
            
            metrics = {"loss": avg_loss, "accuracy": accuracy}
            logger.info(f"Evaluation Metrics (Aggregated on Rank 0): {metrics}")
            
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the model and optimizer state. Only executed by the main process (Rank 0).

        Args:
            path (str): The file path where the checkpoint should be saved.
        """
        if distributed_utils.is_main_process():
            # Unwrap DDP model (.module) to save only the original state_dict
            model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            
            state = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(state, path)
            logger.info(f"Model checkpoint saved by Rank 0 to {path}")
        
        # BARRIER: All processes wait until the main process finishes saving
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the model and optimizer state on all ranks.

        Args:
            path (str): The file path to the checkpoint.
        """
        # Map location ensures the checkpoint is loaded directly onto the current GPU
        map_location = {'cuda:%d' % 0: 'cuda:%d' % distributed_utils.get_rank()} 
        
        # Determine which model object to load the state into (the unwrapped one)
        model_to_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        
        state = torch.load(path, map_location=map_location)
        model_to_load.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded by Rank {distributed_utils.get_rank()}.")

        # BARRIER: All processes wait until all ranks have loaded the checkpoint
        distributed_utils.synchronize_between_processes("load_checkpoint")