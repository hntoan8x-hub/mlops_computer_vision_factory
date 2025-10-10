# cv_factory/shared_libs/ml_core/trainer/implementations/semi_supervised_trainer.py

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.parallel
from shared_libs.ml_core.trainer.base.base_cv_trainer import BaseCVTrainer
from shared_libs.ml_core.trainer.utils import distributed_utils # DDP Utilities

logger = logging.getLogger(__name__)

class SemiSupervisedTrainer(BaseCVTrainer):
    """
    A trainer for semi-supervised learning techniques (e.g., Consistency Regularization), 
    integrated with Distributed Data Parallel (DDP).

    This trainer uses a small amount of labeled data and a large amount
    of unlabeled data, managed across multiple GPUs/nodes.
    """
    
    def __init__(self, model: nn.Module, **kwargs: Dict[str, Any]):
        """
        Initializes the SemiSupervisedTrainer, setting up DDP.

        Args:
            model (nn.Module): The model to be trained.
            **kwargs: Configuration, including 'consistency_loss_fn' and 'consistency_weight'.
        """
        super().__init__(model, **kwargs)
        self.consistency_loss_fn = kwargs.get("consistency_loss_fn", None)
        self.consistency_weight = kwargs.get("consistency_weight", 1.0) # Standard parameter for semi-supervision

        # 1. SETUP DISTRIBUTED ENVIRONMENT
        distributed_utils.setup_distributed_environment() 
        
        # 2. WRAP MODEL FOR DDP
        self.model = distributed_utils.wrap_model_for_distributed(self.model)
        
        if distributed_utils.is_main_process():
            logger.info("SemiSupervisedTrainer initialized and DDP setup.")

    def fit(self, labeled_loader: torch.utils.data.DataLoader, 
            unlabeled_loader: torch.utils.data.DataLoader, 
            val_loader: Optional[torch.utils.data.DataLoader] = None, 
            epochs: int = 10) -> None:
        """
        Trains the model using both labeled and unlabeled data.
        
        Args:
            labeled_loader (DataLoader): DataLoader for labeled data.
            unlabeled_loader (DataLoader): DataLoader for unlabeled data.
            val_loader (Optional[DataLoader]): DataLoader for validation.
            epochs (int): Number of training epochs.
        """
        # Set epoch for DistributedSampler for both loaders
        if distributed_utils.get_world_size() > 1:
            if hasattr(labeled_loader.sampler, 'set_epoch'):
                labeled_loader.sampler.set_epoch(epochs)
            if hasattr(unlabeled_loader.sampler, 'set_epoch'):
                unlabeled_loader.sampler.set_epoch(epochs)
                
        self.model.train()
        
        # Use zip(labeled_loader, unlabeled_loader) to iterate over both simultaneously
        # (Assuming they are correctly setup with compatible lengths/samplers)
        for epoch in range(epochs):
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
                    # Example: get pseudo-labels for consistency loss
                    pseudo_labels = torch.argmax(outputs_ul, dim=1) 
                    consistency_loss = self.consistency_loss_fn(outputs_ul, pseudo_labels)
                    total_loss += consistency_loss * self.consistency_weight
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                if (i + 1) % 100 == 0 and distributed_utils.is_main_process():
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/...], Loss: {total_loss.item():.4f}")

            if val_loader:
                self.evaluate(val_loader)
                
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Evaluates the model on the validation set and aggregates metrics across all processes.
        (Using standard DDP reduction pattern for metrics).
        """
        self.model.eval()
        total_loss = torch.tensor(0.0).to(self.device)
        total = torch.tensor(0).to(self.device)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                total += labels.size(0)

        # AGGREGATION: Reduce/Sum values from all processes
        if distributed_utils.get_world_size() > 1:
            distributed_utils.reduce_tensor_across_processes(total_loss, op=torch.distributed.ReduceOp.SUM)
            distributed_utils.reduce_tensor_across_processes(total, op=torch.distributed.ReduceOp.SUM)
        
        metrics = {}
        if distributed_utils.is_main_process():
            # FINAL CALCULATION ON RANK 0
            avg_loss = total_loss.item() / len(test_loader)
            metrics = {"loss": avg_loss}
            logger.info(f"Evaluation Metrics (Aggregated on Rank 0): {metrics}")
            
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Saves checkpoint, executed by Rank 0 only, ensuring DDP model unwrapping."""
        if distributed_utils.is_main_process():
            model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            state = {'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
            torch.save(state, path)
            logger.info(f"Semi-Supervised model checkpoint saved by Rank 0 to {path}")
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