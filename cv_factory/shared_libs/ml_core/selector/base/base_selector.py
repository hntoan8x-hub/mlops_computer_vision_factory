import abc
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class BaseSelector(abc.ABC):
    """
    Abstract Base Class for all model selectors.

    Defines a standard interface for selecting a model from a list of candidates
    based on predefined criteria.
    """

    @abc.abstractmethod
    def select(self, candidates: List[Dict[str, Any]], **kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Selects the best model from a list of candidates.

        Args:
            candidates (List[Dict[str, Any]]): A list of dictionaries, where each
                                                dictionary represents a candidate model
                                                (e.g., {'name': 'model_A', 'version': 1, 'metrics': {...}}).
            **kwargs: Additional parameters for the selection process.

        Returns:
            Optional[Dict[str, Any]]: The selected model dictionary, or None if no model
                                      meets the selection criteria.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_selection(self, selected_model: Optional[Dict[str, Any]], **kwargs: Dict[str, Any]) -> None:
        """
        Logs the selection decision to a specified logging service.

        Args:
            selected_model (Optional[Dict[str, Any]]): The selected model dictionary, or None.
            **kwargs: Additional logging parameters.
        """
        raise NotImplementedError