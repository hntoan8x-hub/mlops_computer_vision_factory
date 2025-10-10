# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_vit_embedder.py

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple
import pickle

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# <<< CRITICAL: Import the Atomic Logic Class (the Adaptee) >>>
from shared_libs.data_processing.embedders.atomic.vit_embedder import ViTEmbedder 
# We assume ViTEmbedder is the class containing the Hugging Face ViT model logic.

logger = logging.getLogger(__name__)

class CVViTEmbedder(BaseComponent):
    """
    Adapter component for extracting embeddings using Vision Transformers (ViT).
    
    This class is STATEFUL, adheres to BaseComponent, and delegates all model loading, 
    feature extraction setup, and execution to the atomic ViTEmbedder class.
    """
    
    def __init__(self, model_name: str, pretrained: bool = True):
        """
        Initializes the CVViTEmbedder Adapter and the Atomic Embedder.

        Args:
            model_name (str): Identifier for the ViT model (e.g., a Hugging Face ID).
            pretrained (bool): Whether to load pretrained weights.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility)
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Determine device centrally
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        # The atomic class handles the actual Hugging Face model and processor instantiation.
        self.atomic_embedder = ViTEmbedder(
            model_name=self.model_name, 
            pretrained=self.pretrained, 
            device=self.device # Pass device info for model placement
        )
        
        logger.info(f"Initialized CVViTEmbedder Adapter for '{self.model_name}' on {self.device}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVViTEmbedder':
        """
        ViT Embedder is typically stateless for feature extraction inference.
        """
        logger.info("CVViTEmbedder is stateless for transformation, no fitting required.")
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
        logger.info(f"CVViTEmbedder model state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the fitted state (model weights) and re-initializes the atomic embedder via delegation.
        """
        # Delegation of loading the PyTorch model state
        self.atomic_embedder.load(path)
        
        # Ensure the model is still on the correct device after loading
        self.atomic_embedder.model.to(self.device).eval() 
        logger.info(f"CVViTEmbedder model state loaded from {path} and placed on {self.device}.")