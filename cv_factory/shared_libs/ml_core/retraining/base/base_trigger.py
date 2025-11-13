# shared_libs/ml_core/retraining/base/base_trigger.py
import abc
from typing import Dict, Any, Optional
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry # NEW IMPORT

class BaseTrigger(abc.ABC):
    """
    Abstract Base Class for all retraining triggers.

    Defines a standard interface for checking if a retraining job should be initiated
    based on a specific condition.
    """

    def __init__(self, config: Dict[str, Any], registry_service: Optional[BaseRegistry] = None):
        self.config = config
        self.registry = registry_service # NEW: Store the injected registry service

    @abc.abstractmethod
    def check(self, **kwargs: Dict[str, Any]) -> bool:
        """
        Checks if the trigger condition is met.

        Args:
            **kwargs: Additional data required for the check (e.g., current metrics,
                      drift report, timestamp).

        Returns:
            bool: True if the retraining should be triggered, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_reason(self) -> str:
        """
        Returns a string explaining why the trigger was fired.

        Returns:
            str: The reason for the trigger.
        """
        raise NotImplementedError