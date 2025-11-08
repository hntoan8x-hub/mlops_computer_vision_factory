# shared_libs/data_processing/_base/base_embedder.py

import abc
import numpy as np
from typing import Dict, Any, Union, List

# Type hint for input image data (consistent with BaseImageCleaner).
ImageData = Union[np.ndarray, List[np.ndarray]]

# Type hint for the output embedding vectors.
EmbeddingData = Union[np.ndarray, List[np.ndarray]]

class BaseEmbedder(abc.ABC):
    """
    Abstract Base Class for deep learning-based embedding extractors.

    Defines a standard interface for converting preprocessed images into 
    high-dimensional vector representations (embeddings). Embeddings are typically 
    used as input features for downstream classification or clustering models.
    """

    @abc.abstractmethod
    def embed(self, image: ImageData, **kwargs: Dict[str, Any]) -> EmbeddingData:
        """
        Generates embedding vector(s) for the input image(s).

        Args:
            image (ImageData): The input image(s), usually the output from the 
                               Cleaning pipeline. It can be a single np.ndarray 
                               or a list of np.ndarray for batch processing.
            **kwargs: Additional parameters for the embedding model, 
                      e.g., specific layer name for feature extraction.

        Returns:
            EmbeddingData: The generated embedding vector(s). 
                           Typically a 1D or 2D numpy array per image.
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError