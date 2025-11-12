# cv_factory/shared_libs/ml_core/trainer/trainer_factory.py (CẬP NHẬT)

import logging
from typing import Dict, Any, Type, Optional
import torch.nn as nn

from shared_libs.ml_core.trainer.base.base_cv_trainer import BaseCVTrainer
from shared_libs.ml_core.trainer.base.base_trainer import BaseTrainer 
# Import Trainer Implementations
from shared_libs.ml_core.trainer.implementations.cnn_trainer import CNNTrainer
from shared_libs.ml_core.trainer.implementations.finetune_trainer import FinetuneTrainer
from shared_libs.ml_core.trainer.implementations.transformer_trainer import TransformerTrainer 
from shared_libs.ml_core.trainer.implementations.automl_cv_trainer import AutoMLCVTrainer
from shared_libs.ml_core.trainer.implementations.semi_supervised_trainer import SemiSupervisedTrainer
from shared_libs.ml_core.trainer.implementations.contrastive_trainer import ContrastiveTrainer

# CRITICAL NEW IMPORTS: Schema objects 
from shared_libs.ml_core.trainer.configs.trainer_config_schema import TrainerConfig
from shared_libs.ml_core.configs.model_config_schema import ModelConfig

# THÊM IMPORT THIẾT YẾU CHO MLOPS TRACKER
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 

logger = logging.getLogger(__name__)

class TrainerFactory:
    """
    A factory class for creating different types of model trainers.
    
    This class centralizes the creation logic and injects the MLOps Tracker dependency.
    """
    _TRAINER_MAP: Dict[str, Type[BaseTrainer]] = {
        "cnn": CNNTrainer,
        "finetune": FinetuneTrainer,
        "transformer": TransformerTrainer,
        "automl": AutoMLCVTrainer,
        "semi_supervised": SemiSupervisedTrainer,
        "contrastive": ContrastiveTrainer,
    }

    @classmethod
    def create(cls, 
               trainer_type: str, 
               model: Optional[nn.Module], 
               trainer_config: TrainerConfig, 
               model_config: ModelConfig,      
               mlops_tracker: BaseTracker, # <<< THÊM: NHẬN TRACKER >>>
               **kwargs: Dict[str, Any]
              ) -> BaseTrainer:
        """
        Creates and returns a trainer instance, performing Dependency Injection of the MLOps Tracker.
        """
        trainer_cls = cls._TRAINER_MAP.get(trainer_type.lower())
        
        if not trainer_cls:
            supported_trainers = ", ".join(cls._TRAINER_MAP.keys())
            logger.error(f"Unsupported trainer type: '{trainer_type}'. Supported types are: {supported_trainers}")
            raise ValueError(f"Unsupported trainer type: '{trainer_type}'. Supported types are: {supported_trainers}")

        # Xử lý trường hợp Trainer không cần PyTorch Model (ví dụ: AutoML)
        if trainer_cls is AutoMLCVTrainer and model is not None:
             logger.warning(f"Trainer type '{trainer_type}' does not typically use a PyTorch model. Ignoring provided model object.")
             
        logger.info(f"Creating instance of {trainer_cls.__name__} using validated schemas...")
        
        # Truyền MLOps Tracker vào constructor của Trainer
        return trainer_cls(
            model=model, 
            trainer_config=trainer_config,
            model_config=model_config,
            mlops_tracker=mlops_tracker, # <<< TRUYỀN TRACKER VÀO >>>
            **kwargs 
        )