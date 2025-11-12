# shared_libs/ml_core/model_loading/base_model_loading_service.py

import abc
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseModelLoadingService(abc.ABC):
    """
    Abstract Base Class (ABC) for all Model Loading Services.

    Defines the contract for safely locating, downloading, and instantiating 
    a model artifact (e.g., PyTorch, ONNX, Scikit-learn) from a given URI.
    """

    def __init__(self, service_id: str, config: Dict[str, Any]):
        self.service_id = service_id
        self.config = config

    @abc.abstractmethod
    def load(self, model_uri: str, target_device: Optional[str] = None) -> Any:
        """
        Loads the model artifact into memory and returns the instantiated object.

        Args:
            model_uri (str): The artifact URI (e.g., 'models:/name/version', 's3://path').
            target_device (Optional[str]): The target device for model (e.g., 'cuda', 'cpu').

        Returns:
            Any: The instantiated model object (e.g., torch.nn.Module, onnx.Model).

        Raises:
            RuntimeError: If model loading or device placement fails.
        """
        raise NotImplementedError