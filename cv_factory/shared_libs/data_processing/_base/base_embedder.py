import abc
import numpy as np
from typing import Dict, Any, Union, List

# Type hint for embeddings.
EmbeddingData = Union[np.ndarray, List[np.ndarray]]

class BaseEmbedder(abc.ABC):
    """
    Abstract Base Class for deep learning-based embedding extractors.

    Defines a standard interface for converting images into high-dimensional
    vector representations (embeddings).
    """

    @abc.abstractmethod
    def embed(self, image: np.ndarray, **kwargs: Dict[str, Any]) -> EmbeddingData:
        """
        Generates an embedding vector for a single image.

        Args:
            image (np.ndarray): The input image.
            **kwargs: Additional parameters for the embedding model.

        Returns:
            EmbeddingData: The generated embedding vector.
        """
        raise NotImplementedError