import logging
import numpy as np
import torch
from typing import Dict, Any, Union, List

from torchvision import models
from shared_libs.data_processing._base.base_embedder import BaseEmbedder, EmbeddingData

logger = logging.getLogger(__name__)

class CNNEmbedder(BaseEmbedder):
    """
    Extracts embeddings from pre-trained Convolutional Neural Networks (CNNs).
    """

    def __init__(self, model_name: str = "resnet18", pretrained: bool = True):
        """
        Initializes the CNNEmbedder with a pre-trained model.

        Args:
            model_name (str): The name of the CNN model (e.g., "resnet18", "efficientnet_b0").
            pretrained (bool): Whether to use a pre-trained model.
        """
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' is not supported by torchvision.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.__dict__[model_name](pretrained=pretrained)
        
        # Remove the final classification layer to get embeddings
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device).eval()
        
        logger.info(f"Initialized CNNEmbedder with model '{model_name}' on device '{self.device}'.")

    @torch.no_grad()
    def embed(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> EmbeddingData:
        """
        Generates an embedding vector for a single image using the CNN model.

        Args:
            image (np.ndarray): The input image as a NumPy array (H x W x C).
            **kwargs: Additional keyword arguments.

        Returns:
            EmbeddingData: The embedding vector as a NumPy array.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (RGB) NumPy array.")
            
        try:
            # Convert NumPy array to PyTorch tensor and move to device
            tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            tensor_image = tensor_image.to(self.device)
            
            embedding = self.model(tensor_image)
            
            # Squeeze and convert to NumPy array
            embedding_np = embedding.squeeze().cpu().numpy()
            return embedding_np
        
        except Exception as e:
            logger.error(f"Failed to generate embedding for image. Error: {e}")
            raise