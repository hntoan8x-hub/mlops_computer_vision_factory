# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_cnn_embedder.py

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple
import pickle

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# <<< CRITICAL: Import the Atomic Logic Class (the Adaptee) >>>
from shared_libs.data_processing.embedders.atomic.cnn_embedder import CNNEmbedder 
# We assume CNNEmbedder is the class containing the PyTorch model logic.

logger = logging.getLogger(__name__)

class CVCNNEmbedder(BaseComponent):
    """
    Adapter component for extracting embeddings using Convolutional Neural Networks (CNNs).
    
    This class is STATEFUL, adheres to BaseComponent, and delegates all model loading, 
    device management, and embedding execution to the atomic CNNEmbedder class.
    """
    
    def __init__(self, model_name: str, pretrained: bool = True, remove_head: bool = True):
        """
        Initializes the CVCNNEmbedder Adapter and the Atomic Embedder.

        Args:
            model_name (str): Identifier for the CNN model (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained weights.
            remove_head (bool): Whether to remove the final classification layer.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility)
        self.model_name = model_name
        self.pretrained = pretrained
        self.remove_head = remove_head
        
        # Determine device centrally
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        # The atomic class handles the actual PyTorch implementation (e.g., loading torchvision model)
        self.atomic_embedder = CNNEmbedder(
            model_name=self.model_name, 
            pretrained=self.pretrained, 
            remove_head=self.remove_head,
            device=self.device # Pass device info to the atomic layer for placement
        )
        
        logger.info(f"Initialized CVCNNEmbedder Adapter for '{self.model_name}' on {self.device}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVCNNEmbedder':
        """
        Embedder is typically stateless for feature extraction inference.
        """
        logger.info("CVCNNEmbedder is stateless for transformation, no fitting required.")
        return self

    @torch.no_grad()
    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Processes input and extracts embeddings by delegating execution to the atomic embedder.
        """
        # <<< ADAPTER LOGIC: Delegation of transformation >>>
        return self.atomic_embedder.embed(X)
        # End of Adapter Logic

    def save(self, path: str) -> None:
        """
        Saves the fitted state (model weights) of the atomic embedder via delegation.
        """
        # Delegation of saving the PyTorch model state
        self.atomic_embedder.save(path)
        logger.info(f"CVCNNEmbedder model state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the fitted state (model weights) and re-initializes the atomic embedder via delegation.
        """
        # Delegation of loading the PyTorch model state
        self.atomic_embedder.load(path)
        
        # Ensure the model is still on the correct device after loading
        self.atomic_embedder.model.to(self.device).eval() 
        logger.info(f"CVCNNEmbedder model state loaded from {path} and placed on {self.device}.")