import abc
from typing import Dict, Any, Optional

class BaseMLflowService(abc.ABC):
    """
    Abstract Base Class for MLflow interaction services.

    Defines the standard interface for logging experiments, models, 
    and managing the Model Registry across all domains in the CV Factory.
    """

    @abc.abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Logs a single key-value metric to the active MLflow run."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Logs a parameter (e.g., hyperparameters) to the active MLflow run."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs a local file or directory as an artifact to the active run."""
        raise NotImplementedError

    @abc.abstractmethod
    def register_model(self, model_uri: str, name: str) -> str:
        """
        Registers a model artifact with the MLflow Model Registry.

        Args:
            model_uri (str): The URI pointing to the model artifact (e.g., 'runs:/RUN_ID/model').
            name (str): The name under which to register the model.

        Returns:
            str: The version number of the newly registered model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tag_model_version(self, name: str, version: str, tags: Dict[str, str]) -> None:
        """
        [NEW PRODUCTION REQUIREMENT] Tags a specific version of a registered model.

        This is crucial for MLOps tracing (e.g., linking model version to Git SHA, 
        training config hash, or CI build ID).

        Args:
            name (str): The name of the registered model.
            version (str): The model version number.
            tags (Dict[str, str]): Dictionary of tags to apply.
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def transition_model_stage(self, name: str, version: str, stage: str) -> None:
        """
        [NEW PRODUCTION REQUIREMENT] Transitions a registered model version to a new stage.

        This facilitates Continuous Deployment (CD) and production readiness tracking.

        Args:
            name (str): The name of the registered model.
            version (str): The model version number.
            stage (str): The target stage (e.g., 'Staging', 'Production', 'Archived').
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_latest_model_version_uri(self, name: str, stage: str = "Production") -> Optional[str]:
        """
        Retrieves the URI of the latest model version in a specific stage.
        """
        raise NotImplementedError