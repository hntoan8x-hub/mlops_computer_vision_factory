# cv_factory/shared_libs/ml_core/trainer/base/base_cv_trainer.py (CẬP NHẬT)

import abc
import logging
from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from .base_trainer import BaseTrainer
# Import Utilities from their hardened locations
from ..utils import optimizer_utils     
from ..utils import checkpoint_utils    
from ..utils import distributed_utils   

# CRITICAL NEW IMPORTS: Schema validation objects 
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig 
from shared_libs.ml_core.configs.model_config_schema import ModelConfig     

# THÊM IMPORT THIẾT YẾU CHO MLOPS
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker # Import Contract

logger = logging.getLogger(__name__)

class BaseCVTrainer(BaseTrainer, abc.ABC):
    """
    Abstract Base Class for all Computer Vision model trainers.

    Extends BaseTrainer, enforces PyTorch model handling, requires validated configurations,
    và thực hiện Dependency Injection cho MLOps Tracker.
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 trainer_config: TrainerConfig, 
                 model_config: ModelConfig,     
                 mlops_tracker: BaseTracker, # <<< THÊM: DI MLOps TRACKER >>>
                 **kwargs: Dict[str, Any]):
        """
        Initializes the trainer with validated configurations and MLOps services.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            trainer_config (TrainerConfig): Validated Pydantic object for trainer settings.
            model_config (ModelConfig): Validated Pydantic object for model architecture settings.
            mlops_tracker (BaseTracker): The injected service for logging metrics/params.
            **kwargs: Additional optional configurations.
        """
        super().__init__(**kwargs) 
        
        self.model = model
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.mlops_tracker = mlops_tracker # <<< LƯU TRỮ TRACKER >>>
        
        # Determine device based on CUDA availability
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # --- 2. Initialize Optimizer and Scheduler using Utilities ---
        
        optimizer_cfg = trainer_config.optimizer.model_dump() 
        scheduler_cfg = trainer_config.scheduler.model_dump(exclude_none=True) 
        
        self.optimizer: optim.Optimizer = optimizer_utils.get_optimizer(self.model, optimizer_cfg)
        self.scheduler = optimizer_utils.get_scheduler(self.optimizer, scheduler_cfg)
        
        # 3. Initialize Loss Function
        loss_fn_config = trainer_config.params.get("loss_fn_config", {"type": "CrossEntropyLoss"}) if trainer_config.params else {"type": "CrossEntropyLoss"}
        self.loss_fn: nn.Module = self._get_loss_fn(loss_fn_config)

        logger.info(f"Trainer initialized on device: {self.device}. Optimizer: {type(self.optimizer).__name__}")

    def _get_loss_fn(self, config: Dict[str, Any]) -> nn.Module:
        """Helper to create loss function from configuration."""
        loss_type = config.get('type', 'CrossEntropyLoss').lower()
        
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
        [ABSTRACT] Trains the model. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    # --- NEW METHOD: Logging Utility (Sử dụng Tracker đã tiêm) ---
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Logs a metric to the injected MLOps tracker (MLflow) if the current 
        process is the main process (Rank 0).
        """
        if distributed_utils.is_main_process():
            self.mlops_tracker.log_metric(key, value, step)
            
    # --- Checkpoint Management (Using Utilities) ---

    def save(self, path: str, **kwargs: Dict[str, Any]) -> None:
        """Saves the model, optimizer, and scheduler state using checkpoint utilities."""
        
        model_to_save = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        
        state = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'device': self.device.type,
            'trainer_config': self.trainer_config.model_dump(), 
            'model_config': self.model_config.model_dump(),     
            **kwargs
        }
        checkpoint_utils.save_checkpoint(state, path)
        
    def load(self, path: str, **kwargs: Dict[str, Any]) -> None:
        """Loads the model, optimizer, and scheduler state using checkpoint utilities."""
        
        state = checkpoint_utils.load_checkpoint(path)
        
        map_location = self.device if self.device.type == 'cuda' else 'cpu'
        
        model_to_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        
        model_to_load.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if self.scheduler and state.get('scheduler_state_dict'):
             self.scheduler.load_state_dict(state['scheduler_state_dict'])
             
        logging.info(f"Checkpoint loaded to device {self.device}.")