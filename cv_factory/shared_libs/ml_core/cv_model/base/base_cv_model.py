# shared_libs/ml_core/cv_model/base/base_cv_model.py

import abc
from typing import Dict, Any, Optional
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class BaseCVModel(nn.Module, abc.ABC):
    """
    Abstract Base Class for all Computer Vision models in the CV Factory.
    
    Extends torch.nn.Module and enforces standard methods for lifecycle management 
    and configuration inspection.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the model architecture.
        
        Args:
            config (Dict[str, Any]): The model configuration dictionary.
            **kwargs: Additional parameters.
        """
        super().__init__()
        self.config = config
        self.is_pretrained = config.get('pretrained', False)
        
        # NOTE: Model architecture definition should happen in concrete subclasses

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the model configuration dictionary.
        """
        return self.config
        
    def save_architecture(self, path: str) -> None:
        """
        Saves the model architecture definition (e.g., as JSON/YAML).
        """
        try:
            # Simple saving of the config
            import json
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Model architecture config saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model architecture config: {e}")
            raise