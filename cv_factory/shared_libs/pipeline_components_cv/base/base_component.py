import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseComponent(abc.ABC):
    """
    Abstract Base Class for all Computer Vision pipeline components.

    Defines a standard interface for components that can be part of a sequential
    or DAG-based processing pipeline.
    """

    @abc.abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> 'BaseComponent':
        """
        Fits the component to the data. This step is for components that need
        to learn something from the data (e.g., PCA fitting).

        Args:
            X (Any): Input data.
            y (Optional[Any]): Target data.

        Returns:
            BaseComponent: The fitted component instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Applies a transformation to the input data.

        Args:
            X (Any): Input data to be transformed.

        Returns:
            Any: The transformed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Combines fit and transform steps.

        Args:
            X (Any): Input data.
            y (Optional[Any]): Target data.

        Returns:
            Any: The transformed data.
        """
        # A default implementation can be provided to avoid re-writing this for every subclass.
        self.fit(X, y)
        return self.transform(X)

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the state of the component (e.g., learned parameters, model weights) to a file.

        Args:
            path (str): The path to save the state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the state of the component from a file.

        Args:
            path (str): The path to the saved state.
        """
        raise NotImplementedError