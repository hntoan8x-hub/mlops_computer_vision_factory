# shared_libs/ml_core/model_loading/mlflow_model_loading_service.py

import logging
from typing import Any, Dict, Optional
import torch 

from .base_model_loading_service import BaseModelLoadingService
from shared_libs.ml_core.mlflow_service.implementations.mlflow_client_wrapper import MLflowClientWrapper 

logger = logging.getLogger(__name__)

class MLflowModelLoadingService(BaseModelLoadingService):
    """
    Concrete implementation of ModelLoadingService using the MLflow ecosystem.

    This service is responsible for knowing how to interact with MLflow's 
    model tracking and artifact logging to retrieve the model object.
    """

    def __init__(self, service_id: str, config: Dict[str, Any]):
        super().__init__(service_id, config)
        self.mlflow_client = MLflowClientWrapper()

    def load(self, model_uri: str, target_device: Optional[str] = None) -> Any:
        """
        Loads the model artifact using the MLflow client and moves it to the target device.
        """
        logger.info(f"[{self.service_id}] Loading model from URI: {model_uri}")
        
        device = torch.device(target_device or 'cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            model = self.mlflow_client.load_model(model_uri=model_uri)
            
            if isinstance(model, torch.nn.Module):
                model.to(device)
                model.eval() # Set to evaluation mode for inference

            logger.info(f"[{self.service_id}] Model loaded successfully on device: {device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from {model_uri} via MLflow: {e}")
            raise RuntimeError(f"Model loading service failed: {e}")