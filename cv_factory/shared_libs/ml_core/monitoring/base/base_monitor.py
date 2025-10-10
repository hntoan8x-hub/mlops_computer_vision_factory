import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseMonitor(abc.ABC):
    """
    Abstract Base Class for all model monitors.

    Defines a standard interface for checking a specific aspect of a deployed model,
    such as data drift or performance degradation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def check(self, reference_data: Any, current_data: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a check on the model's behavior or data distribution.

        Args:
            reference_data (Any): The baseline data for comparison (e.g., training data distribution).
            current_data (Any): The current data from the deployed model's environment.
            **kwargs: Additional parameters for the check (e.g., model predictions).

        Returns:
            Dict[str, Any]: A report containing the results of the check.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        """
        Determines if the monitor's report triggers an alert.

        Args:
            report (Dict[str, Any]): The report from the `check` method.

        Returns:
            bool: True if an alert should be triggered, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_report_message(self, report: Dict[str, Any]) -> str:
        """
        Generates a human-readable message from the report.

        Args:
            report (Dict[str, Any]): The report from the `check` method.

        Returns:
            str: A formatted message summarizing the findings.
        """
        raise NotImplementedError