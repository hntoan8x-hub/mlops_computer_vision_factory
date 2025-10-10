import logging
from typing import Dict, Any, Type, Optional
import torch.nn as nn

from shared_libs.ml_core.trainer.base.base_cv_trainer import BaseCVTrainer
from shared_libs.ml_core.trainer.implementations.cnn_trainer import CNNTrainer
from shared_libs.ml_core.trainer.implementations.finetune_trainer import FinetuneTrainer
from shared_libs.ml_core.trainer.implementations.transformer_trainer import TransformerTrainer
from shared_libs.ml_core.trainer.implementations.automl_cv_trainer import AutoMLCVTrainer
from shared_libs.ml_core.trainer.implementations.semi_supervised_trainer import SemiSupervisedTrainer
from shared_libs.ml_core.trainer.implementations.contrastive_trainer import ContrastiveTrainer

logger = logging.getLogger(__name__)

class TrainerFactory:
    """
    A factory class for creating different types of model trainers.
    
    This class centralizes the creation logic, allowing for a config-driven
    approach to training pipeline construction.
    """
    _TRAINER_MAP: Dict[str, Type[BaseCVTrainer]] = {
        "cnn": CNNTrainer,
        "finetune": FinetuneTrainer,
        "transformer": TransformerTrainer,
        "automl": AutoMLCVTrainer,
        "semi_supervised": SemiSupervisedTrainer,
        "contrastive": ContrastiveTrainer,
    }

    @classmethod
    def create(cls, trainer_type: str, model: nn.Module, config: Optional[Dict[str, Any]] = None) -> BaseCVTrainer:
        """
        Creates and returns a trainer instance based on its type.

        Args:
            trainer_type (str): The type of trainer to create (e.g., "cnn", "finetune").
            model (nn.Module): The PyTorch model to be trained.
            config (Optional[Dict[str, Any]]): Configuration for the trainer.

        Returns:
            BaseCVTrainer: An instance of the requested trainer.

        Raises:
            ValueError: If the specified trainer_type is not supported.
        """
        config = config or {}
        trainer_cls = cls._TRAINER_MAP.get(trainer_type.lower())
        
        if not trainer_cls:
            supported_trainers = ", ".join(cls._TRAINER_MAP.keys())
            logger.error(f"Unsupported trainer type: '{trainer_type}'. Supported types are: {supported_trainers}")
            raise ValueError(f"Unsupported trainer type: '{trainer_type}'. Supported types are: {supported_trainers}")

        logger.info(f"Creating instance of {trainer_cls.__name__}...")
        return trainer_cls(model=model, **config)