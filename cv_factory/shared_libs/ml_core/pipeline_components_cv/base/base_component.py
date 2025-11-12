# shared_libs/ml_core/pipeline_components_cv/base/base_component.py (FINAL FIX)

import abc
import logging
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class BaseComponent(abc.ABC):
    """
    Abstract Base Class for all Computer Vision pipeline components (Adapters).
    """
    
    # STATIC CONTRACT: Indicates if the component's transform method requires 
    # and processes target data (y). False by default.
    REQUIRES_TARGET_DATA: bool = False 

    @abc.abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> 'BaseComponent':
        """
        Fits the component to the data.

        Args:
            X (Any): Input data.
            y (Optional[Any]): Target data.

        Returns:
            BaseComponent: The fitted component instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    # Sửa: Transform bây giờ chính thức nhận y (Optional)
    def transform(self, X: Any, y: Optional[Any] = None) -> Union[Any, Tuple[Any, Any]]:
        """
        Applies a transformation to the input data (X) and optionally the target data (y).

        Args:
            X (Any): Input data to be transformed.
            y (Optional[Any]): Target data to be transformed.

        Returns:
            Union[Any, Tuple[Any, Any]]: The transformed data (X'), or a tuple (X', Y').
        """
        raise NotImplementedError

    # Sửa: fit_transform KHÔNG còn là abstractmethod
    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Union[Any, Tuple[Any, Any]]:
        """
        Default implementation: fit(X, y) then transform(X, y).
        """
        self.fit(X, y)
        return self.transform(X, y)

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the state of the component (configuration + learned parameters).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the state of the component from a file.
        """
        raise NotImplementedError