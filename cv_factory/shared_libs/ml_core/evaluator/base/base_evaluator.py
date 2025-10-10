import abc
import logging
from typing import Dict, Any, Union, List

logger = logging.getLogger(__name__)

class BaseEvaluator(abc.ABC):
    """
    Abstract Base Class for all model evaluators.

    Defines a standard interface for evaluating a model's performance on a dataset
    and logging the results.
    """

    @abc.abstractmethod
    def evaluate(self, model: Any, data_loader: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the model on the provided data and returns a dictionary of metrics.

        Args:
            model (Any): The machine learning model to be evaluated.
            data_loader (Any): The data source (e.g., PyTorch DataLoader, NumPy array, etc.).
            **kwargs: Additional parameters for evaluation (e.g., 'task_type').

        Returns:
            Dict[str, Any]: A dictionary of computed metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], logger_instance: Any) -> None:
        """
        Logs the computed metrics to a specified logging service (e.g., MLflow, TensorBoard).

        Args:
            metrics (Dict[str, Any]): The dictionary of metrics to log.
            logger_instance (Any): An instance of the logger service client.
        """
        raise NotImplementedError