# cv_factory/shared_libs/ml_core/trainer/implementations/depth_trainer.py

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp 

from shared_libs.ml_core.trainer.base.base_distributed_trainer import BaseDistributedTrainer
from shared_libs.ml_core.trainer.utils import distributed_utils 
# IMPORT THE EVALUATION ORCHESTRATOR
from shared_libs.ml_core.evaluator.evaluation_orchestrator import EvaluationOrchestrator 

from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

logger = logging.getLogger(__name__)

class DepthTrainer(BaseDistributedTrainer): 
    """
    Concrete trainer for Depth Estimation models (Regression Task), 
    with support for Distributed Data Parallel (DDP), Mixed Precision, and integrated 
    Evaluation Orchestration for comprehensive metric reporting.
    """

    def __init__(self, 
                 model: nn.Module,
                 trainer_config: TrainerConfig,
                 model_config: ModelConfig,     
                 **kwargs: Dict[str, Any]):
        """
        Initializes the Depth Trainer, GradScaler, and the Evaluation Orchestrator.
        """
        super().__init__(model, trainer_config, model_config, **kwargs)
        
        # 1. SETUP MIXED PRECISION SCALER
        if self.device.type == 'cuda':
             self.scaler = amp.GradScaler()
             logger.info("Mixed Precision (AMP) GradScaler initialized for DepthTrainer.")
        else:
             self.scaler = None
             
        self.max_grad_norm = self.trainer_config.params.get('max_grad_norm', None)
        self.grad_accumulation_steps = self.trainer_config.params.get('grad_accumulation_steps', 1)

        # 2. EVALUATION ORCHESTRATOR SETUP (Tích hợp Evaluator)
        # Lấy cấu hình Evaluator từ trainer_config.params hoặc kwargs
        evaluator_config = kwargs.get("evaluator_config", {})
        if not evaluator_config:
             logger.warning("Evaluator config not provided. Metrics reporting will be limited to loss.")
        
        # NOTE: Giả định evaluator_config chứa 'task_type' (depth_estimation) và 'metrics'
        self.evaluator = EvaluationOrchestrator(config=evaluator_config)
            
        if distributed_utils.is_main_process():
            logger.info(f"DepthTrainer initialized. Loss Fn: {type(self.loss_fn).__name__}")
            

    # --- CORE TRAINING LOOP IMPLEMENTATION ---

    def train_step(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Performs a single distributed training step for Depth Estimation, 
        including Mixed Precision (AMP) and Gradient Accumulation logic.
        """
        self.model.train()
        
        inputs = data['image'].to(self.device)
        targets = data['depth_map'].to(self.device).float()
        
        if targets.ndim == 3:
            targets = targets.unsqueeze(1) 

        # 1. MIXED PRECISION: Autocast
        with amp.autocast(enabled=(self.device.type == 'cuda')):
            outputs = self.model(inputs) 
            
            if isinstance(outputs, dict) and 'pred_depth' in outputs:
                pred_depth = outputs['pred_depth']
            elif isinstance(outputs, torch.Tensor):
                pred_depth = outputs
            else:
                raise TypeError("Model output format not recognized for Depth Estimation.")
            
            loss = self.loss_fn(pred_depth, targets) / self.grad_accumulation_steps

        # 2. BACKWARD PASS & SCALING
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {"loss": loss.item() * self.grad_accumulation_steps}


    def _run_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader], 
                   epoch: int, 
                   total_epochs: int) -> None:
        """Runs the core training loop for one epoch with optimization steps."""
        
        self.model.train()
        
        for i, data in enumerate(train_loader):
            step_metrics = self.train_step(data)
            
            # 3. OPTIMIZER STEP (Gradient Accumulation Logic)
            if (i + 1) % self.grad_accumulation_steps == 0:
                # Apply Gradient Clipping if configured
                if self.max_grad_norm is not None:
                    if self.scaler:
                         self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                         
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update() 
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad() 

            if (i + 1) % 100 == 0 and distributed_utils.is_main_process():
                logger.info(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/...], Loss: {step_metrics['loss']:.4f}")

        # Handle final optimization step if accumulation steps were not met
        if len(train_loader) % self.grad_accumulation_steps != 0:
            if distributed_utils.is_main_process():
                logger.debug("Running final optimization step for accumulated gradients.")
            
            if self.max_grad_norm is not None:
                if self.scaler:
                     self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.scheduler:
            self.scheduler.step()

        if val_loader:
            self.evaluate(val_loader)
            
    # --- CORE EVALUATION (Using Orchestrator) ---
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluates the Depth Estimation model by delegating the process to the 
        EvaluationOrchestrator to compute comprehensive depth metrics (RMSE, Delta_n, etc.).
        """
        # Chỉ Rank 0 mới chạy Evaluation và tính toán Metric
        if distributed_utils.is_main_process():
            logger.info("Starting comprehensive evaluation on Rank 0...")
            
            # Gọi EvaluationOrchestrator để tính toàn bộ metrics đã cấu hình
            results = self.evaluator.evaluate(self.model, data_loader)
            
            # Log kết quả ra console
            logger.info(f"Evaluation Results: {results['metrics']}")
            
            return results['metrics']
        
        # Các rank khác chờ đồng bộ
        distributed_utils.synchronize_between_processes("evaluation_finish")
        return {} # Các rank khác trả về Dict rỗng


    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
         # Defined above
         return self.train_step(*args, **kwargs) 

    # --- CHECKPOINTING (DELEGATED TO PARENT) ---
    def save_checkpoint(self, path: str) -> None:
        """Saves checkpoint, executed by Rank 0 only, including the GradScaler state."""
        if distributed_utils.is_main_process():
            if self.scaler:
                 # Pass scaler state dict to the BaseCVTrainer.save method
                 super().save(path, scaler_state_dict=self.scaler.state_dict())
            else:
                 super().save(path)
        distributed_utils.synchronize_between_processes("save_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads checkpoint across all ranks, including the GradScaler state."""
        # super().load returns the state dictionary loaded from checkpoint_utils
        state = super().load(path) 
        if self.scaler and state.get('scaler_state_dict'):
             self.scaler.load_state_dict(state['scaler_state_dict'])
        distributed_utils.synchronize_between_processes("load_checkpoint")