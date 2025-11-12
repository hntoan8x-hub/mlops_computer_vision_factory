# cv_factory/shared_libs/ml_core/pipeline_components_cv/_embedder/cv_cnn_embedder.py (FIXED)

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.embedders.atomic.cnn_embedder import CNNEmbedder 

logger = logging.getLogger(__name__)

class CVCNNEmbedder(BaseComponent):
    """
    Adapter component for extracting embeddings using Convolutional Neural Networks (CNNs).
    
    This class is STATEFUL (Persistence) as it manages model weights, but Stateless (ML) 
    as fit is a no-op for feature extraction.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, model_name: str, pretrained: bool = True, remove_head: bool = True):
        """
        Initializes the CVCNNEmbedder Adapter and the Atomic Embedder.

        Args:
            model_name (str): Identifier for the CNN model (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained weights.
            remove_head (bool): Whether to remove the final classification layer.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.remove_head = remove_head
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.atomic_embedder = CNNEmbedder(
            model_name=self.model_name, 
            pretrained=self.pretrained, 
            remove_head=self.remove_head,
            device=self.device
        )
        
        logger.info(f"Initialized CVCNNEmbedder Adapter for '{self.model_name}' on {self.device}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVCNNEmbedder':
        """
        Embedder is typically stateless for feature extraction inference.
        
        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVCNNEmbedder: The component instance.
        """
        logger.info("CVCNNEmbedder is stateless for transformation, no fitting required.")
        return self

    @torch.no_grad()
    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Any] = None) -> np.ndarray:
        """
        Processes input and extracts embeddings by delegating execution to the atomic embedder.
        
        Args:
            X (Union[np.ndarray, torch.Tensor]): The input image data.
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The extracted feature embeddings.
        """
        return self.atomic_embedder.embed(X)

    def save(self, path: str) -> None:
        """Saves the fitted state (model weights) of the atomic embedder via delegation."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.atomic_embedder.save(path)
        logger.info(f"CVCNNEmbedder model state saved to {path}.")

    def load(self, path: str) -> None:
        """Loads the fitted state (model weights) and re-initializes the atomic embedder via delegation."""
        self.atomic_embedder.load(path)
        
        # Ensure the model is still on the correct device after loading
        self.atomic_embedder.model.to(self.device).eval() 
        logger.info(f"CVCNNEmbedder model state loaded from {path} and placed on {self.device}.")