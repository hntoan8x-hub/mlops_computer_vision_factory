# shared_libs/data_processing/image_components/embedders/atomic/cnn_embedder.py

import logging
import numpy as np
import torch
from typing import Dict, Any, Union, List

from torchvision import models
from shared_libs.data_processing._base.base_embedder import BaseEmbedder, EmbeddingData, ImageData

logger = logging.getLogger(__name__)

class CNNEmbedder(BaseEmbedder):
    """
    Extracts embeddings from pre-trained Convolutional Neural Networks (CNNs).

    This component is the atomic logic layer and handles model loading, device 
    placement, and batch inference. It implements save/load for MLOps compliance.
    """

    def __init__(self, model_name: str = "resnet18", pretrained: bool = True):
        """
        Initializes the CNNEmbedder with a pre-trained model.

        Args:
            model_name (str): The name of the CNN model (e.g., "resnet18", "efficientnet_b0").
            pretrained (bool): Whether to use a pre-trained model.
        
        Raises:
            ValueError: If the model name is not supported by torchvision.
        """
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' is not supported by torchvision.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.__dict__[model_name](pretrained=pretrained)
        
        # Remove the final classification layer to get embeddings
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device).eval()
        
        logger.info(f"Initialized CNNEmbedder with model '{model_name}' on device '{self.device}'.")

    # --- MLOps Lifecycle Methods ---
    
    def save(self, path: str) -> None:
        """
        Saves the state dictionary of the underlying PyTorch model.

        Args:
            path (str): The file path where the model state will be saved.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"CNNEmbedder model state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the state dictionary and sets the model to evaluation mode.

        Args:
            path (str): The file path from which the model state will be loaded.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device).eval()
        logger.info(f"CNNEmbedder model state loaded from {path}.")

    # --- Core Logic ---
    
    @torch.no_grad()
    def embed(self, image: ImageData, **kwargs: Dict[str, Any]) -> EmbeddingData:
        """
        Generates embedding vector(s) for a single image or a list of images (batch).

        Args:
            image (ImageData): The input image(s) as NumPy array(s) (H x W x C).
            **kwargs: Additional keyword arguments.

        Returns:
            EmbeddingData: The embedding vector(s) as NumPy array(s) or a list of NumPy arrays.
        
        Raises:
            TypeError: If input type is unsupported.
            ValueError: If image dimensions are incorrect.
        """
        if isinstance(image, np.ndarray):
            image_list = [image]
            is_batch = False
        elif isinstance(image, list):
            image_list = image
            is_batch = True
        else:
            raise TypeError("Input must be a NumPy array or a list of NumPy arrays.")
            
        if not image_list:
            return np.array([]) if not is_batch else []

        try:
            # Stack images into a single PyTorch batch tensor
            tensors = []
            for img in image_list:
                if len(img.shape) != 3 or img.shape[2] != 3:
                     raise ValueError("Input image must be a 3-channel (RGB) NumPy array.")
                tensors.append(torch.from_numpy(img).permute(2, 0, 1).float())

            batch_tensor = torch.stack(tensors).to(self.device)
            
            embedding_batch = self.model(batch_tensor)
            
            # Convert to NumPy array
            embedding_np = embedding_batch.squeeze().cpu().numpy()
            
            # Handle single vs. batch output format
            if not is_batch and embedding_np.ndim > 1:
                 # If input was single but batch result is 2D (due to squeeze), take the first row
                 return embedding_np[0] 
            elif not is_batch and embedding_np.ndim == 1:
                 # If input was single and result is 1D (vector), return the vector
                 return embedding_np
            
            # If input was a batch, return the batch result (List of vectors)
            return embedding_np.tolist() if embedding_np.ndim > 1 else [embedding_np]

        except Exception as e:
            logger.error(f"Failed to generate embedding batch. Error: {e}")
            raise