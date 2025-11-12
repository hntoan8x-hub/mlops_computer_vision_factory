# cv_factory/shared_libs/ml_core/mlflow_service/mlflow_service.py

import logging
from typing import Dict, Any, Optional, List
import torch.nn as nn

# Import Factories and Config
from shared_libs.ml_core.mlflow_service.factories.tracker_factory import TrackerFactory
from shared_libs.ml_core.mlflow_service.factories.registry_factory import RegistryFactory
from shared_libs.ml_core.mlflow_service.configs.mlflow_config_schema import MLflowConfig # Dùng schema đã validated
from shared_libs.ml_core.mlflow_service.implementations.mlflow_client_wrapper import MLflowClientWrapper

# Import Base Contracts
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceError

logger = logging.getLogger(__name__)

class MLflowService:
    """
    MLflow Service Façade.

    This class orchestrates all MLOps tracking and model registry operations,
    providing a simple, technology-agnostic interface to the rest of the application.
    """

    def __init__(self, config: MLflowConfig):
        """
        Initializes the service by setting up the MLflow client and loading
        the Tracker and Registry implementations via Factories.
        """
        self.config = config
        self.tracking_uri = config.tracking_uri

        # 1. Initialize Client Wrapper (handles Singleton and URI setup)
        # MLflowClientWrapper ensures connection and retries.
        self.client_wrapper = MLflowClientWrapper(tracking_uri=self.tracking_uri)
        self.mlflow_client = self.client_wrapper.client # Get the underlying MlflowClient

        # 2. Initialize Tracker via Factory
        tracker_type = config.tracker.type
        self.tracker: BaseTracker = TrackerFactory.create(
            tracker_type=tracker_type,
            config={"tracking_uri": self.tracking_uri, **config.tracker.params}
        )
        
        # 3. Initialize Registry via Factory
        registry_type = config.registry.type
        self.registry: BaseRegistry = RegistryFactory.create(
            registry_type=registry_type,
            config={"tracking_uri": self.tracking_uri, **config.registry.params}
        )
        
        logger.info(f"MLflowService initialized with Tracker: {type(self.tracker).__name__}, Registry: {type(self.registry).__name__}")


    # --- Tracking Methods (Delegated to self.tracker, called by Trainer/Evaluator) ---

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """Starts a new experiment run."""
        return self.tracker.start_run(run_name)

    def end_run(self, status: str = "FINISHED") -> None:
        """Ends the current run."""
        self.tracker.end_run(status)

    def log_param(self, key: str, value: Any) -> None:
        """Logs a single parameter."""
        self.tracker.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Logs a single metric (used by Trainer/Evaluator)."""
        # Thêm step nếu cần
        if step is not None and hasattr(self.tracker, 'log_metric_with_step'):
             self.tracker.log_metric_with_step(key, value, step)
        else:
             self.tracker.log_metric(key, value)

    def log_model(self, model: nn.Module, artifact_path: str) -> None:
        """Logs the trained model (used by Trainer)."""
        self.tracker.log_model(model, artifact_path)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs a local file as an artifact."""
        self.tracker.log_artifact(local_path, artifact_path)

    # --- Registry & Search Methods (Delegated to self.registry/self.client) ---
    
    def search_runs(self, experiment_ids: List[str], filter_string: str = "", order_by: List[str] = ["metrics.mAP DESC"]) -> List[Dict[str, Any]]:
        """
        Searches MLflow runs for candidate models (used by Selector).

        Returns run data structured for the Selector.
        """
        try:
            # Use the underlying client for search functionality
            runs = self.mlflow_client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                order_by=order_by,
                output_type="SEARCH_RUNS"
            )
            
            # NOTE: Logic chuyển đổi MLflow Run object sang Dict Candidate (cho Selector)
            candidates = []
            for run in runs:
                candidate = {
                    "name": run.data.tags.get("mlflow.runName", run.info.run_id),
                    "run_id": run.info.run_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "artifact_uri": run.info.artifact_uri
                }
                candidates.append(candidate)
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to search MLflow runs: {e}")
            raise MLflowServiceError(f"Search runs failed: {e}")

    def register_model_version(self, model_name: str, run_id: str, artifact_path: str, description: Optional[str] = None) -> Any:
        """Registers a model from a run (used after Selector confirms the best model)."""
        return self.registry.register_model(model_name, run_id, artifact_path, description)

    def transition_stage(self, model_name: str, version: int, new_stage: str) -> None:
        """Transitions a model version to a new stage."""
        self.registry.transition_model_stage(model_name, version, new_stage)
        
    def get_model_uri_by_stage(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """Gets the URI to load the model currently in a specific stage."""
        return self.registry.get_model_uri(model_name, stage)