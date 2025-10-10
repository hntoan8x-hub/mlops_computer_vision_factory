import logging
import numpy as np
import torch
from typing import Dict, Any, Union, List

from transformers import ViTFeatureExtractor, ViTModel
from shared_libs.data_processing._base.base_embedder import BaseEmbedder, EmbeddingData

logger = logging.getLogger(__name__)

class ViTEmbedder(BaseEmbedder):
    """
    Extracts embeddings from a pre-trained Vision Transformer (ViT).
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initializes the ViTEmbedder with a pre-trained model.

        Args:
            model_name (str): The name of the ViT model from Hugging Face (e.g., "google/vit-base-patch16-224").
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        
        logger.info(f"Initialized ViTEmbedder with model '{model_name}' on device '{self.device}'.")

    @torch.no_grad()
    def embed(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> EmbeddingData:
        """
        Generates an embedding vector for a single image using the ViT model.

        Args:
            image (np.ndarray): The input image as a NumPy array (H x W x C).
            **kwargs: Additional keyword arguments.

        Returns:
            EmbeddingData: The embedding vector as a NumPy array.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (RGB) NumPy array.")

        try:
            # Preprocess the image using the feature extractor
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get the model output
            outputs = self.model(**inputs)
            
            # The 'last_hidden_state' of the [CLS] token is commonly used as the embedding
            embedding = outputs.last_hidden_state[:, 0, :]
            
            embedding_np = embedding.squeeze().cpu().numpy()
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to generate embedding with ViT model. Error: {e}")
            raise