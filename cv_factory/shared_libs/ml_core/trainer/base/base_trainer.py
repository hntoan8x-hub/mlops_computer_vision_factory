import abc
from typing import Dict, Any, Optional

class BaseTrainer(abc.ABC):
    """
    Abstract Base Class for all machine learning model trainers.

    Defines a standard interface for the core training loop, evaluation, and
    model persistence, regardless of the specific task or model.
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Trains the model on the provided data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Performs a single training step (e.g., one batch).
        
        Returns:
            Dict[str, Any]: A dictionary of metrics for the step.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluates the model on the provided data.
        
        Returns:
            Dict[str, Any]: A dictionary of evaluation metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Saves the model and/or trainer state.
        
        Args:
            path (str): The path to save the state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str, **kwargs) -> None:
        """
        Loads the model and/or trainer state.
        
        Args:
            path (str): The path to the saved state.
        """
        raise NotImplementedError