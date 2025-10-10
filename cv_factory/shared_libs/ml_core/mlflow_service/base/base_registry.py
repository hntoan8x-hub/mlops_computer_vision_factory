import abc
from typing import Dict, Any, List, Optional

class BaseRegistry(abc.ABC):
    """
    Abstract Base Class for all model registries.

    Defines a standard interface for registering, versioning, and retrieving
    models from a central registry.
    """

    @abc.abstractmethod
    def register_model(self, model_name: str, run_id: str, artifact_path: str, description: Optional[str] = None) -> Any:
        # ... (Method body remains the same)
        raise NotImplementedError

    # --- NEW PRODUCTION METHOD ---
    @abc.abstractmethod
    def tag_model_version(self, model_name: str, version: int, tags: Dict[str, str]) -> None:
        """
        Tags a specific version of a registered model with custom metadata.

        This is crucial for MLOps tracing (linking model version to Git SHA, CI Build ID, etc.).

        Args:
            model_name (str): The name of the registered model.
            version (int): The model version number.
            tags (Dict[str, str]): Dictionary of tags to apply (key-value pairs).
        """
        raise NotImplementedError
    # -----------------------------

    @abc.abstractmethod
    def get_latest_version(self, model_name: str, stage: str = "Production") -> Optional[Any]:
        # ... (Method body remains the same)
        raise NotImplementedError

    @abc.abstractmethod
    def transition_model_stage(self, model_name: str, version: int, new_stage: str) -> None:
        """
        Transitions a specific model version to a new stage.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_uri(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """
        Gets the URI of a model version, useful for loading the model.
        """
        raise NotImplementedError