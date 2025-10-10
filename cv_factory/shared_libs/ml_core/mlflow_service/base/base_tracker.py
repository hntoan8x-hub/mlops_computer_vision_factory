import abc
from typing import Dict, Any, Union, Optional

class BaseTracker(abc.ABC):
    """
    Abstract Base Class for all ML experiment trackers.

    Defines a standard interface for logging metrics, parameters, and artifacts
    for an experiment run.
    """

    @abc.abstractmethod
    def start_run(self, run_name: Optional[str] = None) -> Any:
        """
        Starts a new experiment run.

        Args:
            run_name (Optional[str]): The name of the run.

        Returns:
            Any: The run object or ID.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """
        Ends the current run.

        Args:
            status (str): The status of the run (e.g., "FINISHED", "FAILED").
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """
        Logs a single parameter for the current run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Logs a dictionary of parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_metric(self, key: str, value: float) -> None:
        """
        Logs a single metric for the current run.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Logs a dictionary of metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Logs a local file as an artifact.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> None:
        """
        Logs a model as a special artifact.
        """
        raise NotImplementedError