import abc
from typing import Dict, Any

class BaseRetrainOrchestrator(abc.ABC):
    """
    Abstract Base Class for all retraining orchestrators.

    Defines a standard interface for managing the end-to-end retraining process.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def run(self, **kwargs: Dict[str, Any]) -> None:
        """
        Executes the retraining job. This method should handle the full workflow
        from data preparation to model registration.

        Args:
            **kwargs: Arguments needed to run the retraining job.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_job_status(self, status: str, **kwargs: Dict[str, Any]) -> None:
        """
        Logs the status of the retraining job to a logging/monitoring service.

        Args:
            status (str): The status of the job (e.g., 'started', 'completed', 'failed').
        """
        raise NotImplementedError