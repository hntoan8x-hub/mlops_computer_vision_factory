import numpy as np
import logging
from typing import Union, List

logger = logging.getLogger(__name__)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalizes a batch of embeddings to have unit L2 norm.
    This is often required for cosine similarity search.

    Args:
        embeddings (np.ndarray): A 2D NumPy array of embedding vectors.

    Returns:
        np.ndarray: The L2-normalized embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    logger.info("Embeddings normalized.")
    return normalized

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculates the cosine similarity between two vectors or batches of vectors.

    Args:
        a (np.ndarray): The first vector(s).
        b (np.ndarray): The second vector(s).

    Returns:
        Union[float, np.ndarray]: The cosine similarity score(s).
    """
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm.T)