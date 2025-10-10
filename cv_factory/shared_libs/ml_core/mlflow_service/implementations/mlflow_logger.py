import logging
from typing import Dict, Any, Union, Optional
import mlflow
import torch

from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.mlflow_service.implementations.mlflow_client_wrapper import MLflowClientWrapper
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceError

logger = logging.getLogger(__name__)

class MLflowLogger(BaseTracker):
    """
    Concrete implementation of BaseTracker for MLflow.
    """
    def __init__(self, tracking_uri: Optional[str] = None):
        self.client_wrapper = MLflowClientWrapper(tracking_uri=tracking_uri)

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        try:
            active_run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run with ID: {active_run.info.run_id}")
            return active_run
        except Exception as e:
            raise MLflowServiceError(f"Failed to start MLflow run: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        try:
            mlflow.end_run(status)
            logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            raise MLflowServiceError(f"Failed to end MLflow run: {e}")

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float) -> None:
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model: torch.nn.Module, artifact_path: str) -> None:
        # Use mlflow.pytorch.log_model for PyTorch models
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path)
        logger.info(f"Model logged as artifact at '{artifact_path}'.")