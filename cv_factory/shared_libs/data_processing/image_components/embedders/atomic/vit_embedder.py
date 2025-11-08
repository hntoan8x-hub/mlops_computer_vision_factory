# shared_libs/data_processing/image_components/embedders/atomic/vit_embedder.py

import logging
import numpy as np
import torch
from typing import Dict, Any, Union, List

from transformers import ViTFeatureExtractor, ViTModel 
from shared_libs.data_processing._base.base_embedder import BaseEmbedder, EmbeddingData, ImageData

logger = logging.getLogger(__name__)

class ViTEmbedder(BaseEmbedder):
    """
    Extracts embeddings from a pre-trained Vision Transformer (ViT).

    Handles model loading, device placement, batch inference, and MLOps state management 
    for Hugging Face models.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initializes the ViTEmbedder with a pre-trained model.

        Args:
            model_name (str): The name of the ViT model from Hugging Face (e.g., "google/vit-base-patch16-224").
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Hugging Face components
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        
        logger.info(f"Initialized ViTEmbedder with model '{model_name}' on device '{self.device}'.")

    # --- MLOps Lifecycle Methods ---
    
    def save(self, path: str) -> None:
        """
        Saves the Hugging Face model and feature extractor state to a directory.

        Args:
            path (str): The directory path where the state will be saved.
        """
        # Hugging Face models often require saving the entire model structure
        self.model.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)
        logger.info(f"ViTEmbedder model and feature extractor state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the model and feature extractor from a saved path.

        Args:
            path (str): The directory path from which the state will be loaded.
        """
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(path)
        self.model = ViTModel.from_pretrained(path)
        self.model.to(self.device).eval()
        logger.info(f"ViTEmbedder model state loaded from {path}.")


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
            # Batch processing using ViTFeatureExtractor (handles preprocessing and batching)
            inputs = self.feature_extractor(images=image_list, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # The embedding is typically the 'last_hidden_state' of the [CLS] token
            embedding_batch = outputs.last_hidden_state[:, 0, :]
            
            embedding_np = embedding_batch.cpu().numpy()
            
            # Handle single vs. batch output format
            if not is_batch:
                 return embedding_np[0] 
            
            return embedding_np.tolist() if embedding_np.ndim > 1 else [embedding_np]
            
        except Exception as e:
            logger.error(f"Failed to generate embedding batch with ViT model. Error: {e}")
            raise