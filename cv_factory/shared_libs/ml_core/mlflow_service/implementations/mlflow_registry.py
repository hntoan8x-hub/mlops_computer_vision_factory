import logging
from typing import Dict, Any, List, Optional
import mlflow

from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
from shared_libs.ml_core.mlflow_service.implementations.mlflow_client_wrapper import MLflowClientWrapper
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceError

logger = logging.getLogger(__name__)

class MLflowRegistry(BaseRegistry):
    """
    Concrete implementation of BaseRegistry for MLflow Model Registry.
    """
    def __init__(self, tracking_uri: Optional[str] = None):
        self.client_wrapper = MLflowClientWrapper(tracking_uri=tracking_uri)
        self.client = self.client_wrapper.client

    def register_model(self, model_name: str, run_id: str, artifact_path: str, description: Optional[str] = None) -> Any:
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            registered_model = self.client.create_registered_model(model_name)
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=description
            )
            logger.info(f"Registered model '{model_name}' (version {model_version.version}) from run '{run_id}'.")
            return model_version
        except Exception as e:
            raise MLflowServiceError(f"Failed to register model: {e}")

    def get_latest_version(self, model_name: str, stage: str = "Production") -> Optional[Any]:
        try:
            return self.client.get_latest_versions(model_name, stages=[stage])
        except Exception as e:
            logger.error(f"Failed to get latest version for model '{model_name}' at stage '{stage}': {e}")
            return None

    def transition_model_stage(self, model_name: str, version: int, new_stage: str) -> None:
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=new_stage
            )
            logger.info(f"Transitioned model '{model_name}' version {version} to stage '{new_stage}'.")
        except Exception as e:
            raise MLflowServiceError(f"Failed to transition model stage: {e}")

    def get_model_uri(self, model_name: str, stage: str = "Production") -> Optional[str]:
        try:
            latest_version = self.client.get_latest_versions(model_name, stages=[stage])[0]
            # Assumes the run ID and artifact path can be retrieved from the version info
            return f"runs:/{latest_version.run_id}/{latest_version.source.split('/')[-1]}"
        except Exception as e:
            logger.error(f"Failed to get model URI for '{model_name}' at stage '{stage}': {e}")
            return None