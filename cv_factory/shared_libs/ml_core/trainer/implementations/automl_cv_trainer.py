# cv_factory/shared_libs/ml_core/trainer/implementations/automl_cv_trainer.py

import logging
import pandas as pd
from typing import Dict, Any, Optional
from flaml import AutoML
from sklearn.metrics import accuracy_score

from shared_libs.ml_core.trainer.base.base_cv_trainer import BaseCVTrainer
from shared_libs.ml_core.trainer.utils import distributed_utils # DDP Utilities

logger = logging.getLogger(__name__)

class AutoMLCVTrainer(BaseCVTrainer):
    """
    Concrete trainer that uses an AutoML framework (e.g., FLAML) for model training.
    
    Since AutoML searches are computationally expensive, execution is restricted 
    to the main process (Rank 0) if running in a distributed environment.
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Initializes the AutoMLCVTrainer. Note: BaseCVTrainer's model/device setup is bypassed.
        """
        # Call DDP setup first, even if we don't use DDP, to establish Rank
        distributed_utils.setup_distributed_environment() 
        
        if distributed_utils.is_main_process():
            self.automl_model = AutoML(**kwargs)
            self.best_model = None
            self.config = kwargs
            logger.info("AutoMLCVTrainer initialized on Rank 0.")
        else:
            self.automl_model = None
            self.best_model = None
            self.config = kwargs
            logger.info(f"AutoMLCVTrainer skipped initialization on Rank {distributed_utils.get_rank()}.")

    def fit(self, train_loader: Any, val_loader: Optional[Any] = None, **kwargs: Dict[str, Any]) -> None:
        """
        Trains the AutoML model. Only executed by the main process (Rank 0).
        """
        if not distributed_utils.is_main_process():
            distributed_utils.synchronize_between_processes("automl_fit_wait")
            return

        logger.info("Starting AutoML training process (Rank 0 only)...")
        
        if self.automl_model is None:
            raise RuntimeError("AutoML model is not initialized. Run on Rank 0.")
            
        X_train = kwargs.get("X_train")
        y_train = kwargs.get("y_train")
        
        if X_train is None or y_train is None:
            raise ValueError("AutoML Trainer requires X_train and y_train in kwargs.")
            
        self.automl_model.fit(X_train=X_train, y_train=y_train)
        self.best_model = self.automl_model.best_model_for_estimator(self.automl_model.best_estimator)
        
        logger.info(f"AutoML training finished. Best model found: {self.automl_model.best_estimator}")
        
        # Synchronize: Signal other processes that Rank 0 has finished the expensive search
        distributed_utils.synchronize_between_processes("automl_fit_complete")


    def evaluate(self, test_loader: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the best model found by the AutoML framework (Rank 0 only).
        """
        if not distributed_utils.is_main_process():
            return {} # Only Rank 0 returns metrics

        if self.best_model is None:
            raise RuntimeError("Cannot evaluate. No model has been trained yet.")
            
        X_test = kwargs.get("X_test")
        y_test = kwargs.get("y_test")
        
        if X_test is None or y_test is None:
            raise ValueError("AutoML Trainer requires X_test and y_test in kwargs for evaluation.")
            
        predictions = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        metrics = {"accuracy": accuracy}
        logger.info(f"Evaluation Metrics for best AutoML model: {metrics}")
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Saves the AutoML model (Rank 0 only)."""
        if distributed_utils.is_main_process():
            if self.automl_model:
                self.automl_model.save(path)
                logger.info(f"AutoML model checkpoint saved to {path}")
            else:
                logger.warning("Attempted to save AutoML model, but it was not initialized on this rank.")
        distributed_utils.synchronize_between_processes("save_automl_checkpoint")

    def load_checkpoint(self, path: str) -> None:
        """Loads the AutoML model (Rank 0 only)."""
        if distributed_utils.is_main_process():
            if self.automl_model:
                self.automl_model.load(path)
                self.best_model = self.automl_model.best_model_for_estimator(self.automl_model.best_estimator)
                logger.info(f"AutoML model checkpoint loaded from {path}")
            else:
                 logger.warning("Attempted to load AutoML model, but it was not initialized on this rank.")
        distributed_utils.synchronize_between_processes("load_automl_checkpoint")