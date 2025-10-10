import abc
from typing import Dict, Any, Optional

class BaseReporter(abc.ABC):
    """
    Abstract Base Class for all reporting services.

    Defines a standard interface for pushing monitoring reports and metrics
    to various destinations.
    """

    @abc.abstractmethod
    def report(self, report_name: str, report_data: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        """
        Reports a monitoring outcome to a specific service.

        Args:
            report_name (str): The name of the report (e.g., "data_drift_report").
            report_data (Dict[str, Any]): The data to be reported.
            **kwargs: Additional parameters for the reporting service.
        """
        raise NotImplementedError