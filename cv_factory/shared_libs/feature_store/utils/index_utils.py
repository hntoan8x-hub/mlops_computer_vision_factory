import logging
import faiss
import numpy as np

logger = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a basic FAISS index from a set of embeddings.
    
    Args:
        embeddings (np.ndarray): A 2D NumPy array of embeddings.
        
    Returns:
        faiss.Index: The constructed FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"Built FAISS index with {index.ntotal} vectors and dimension {dim}.")
    return index

def persist_index(index: faiss.Index, path: str) -> None:
    """
    Persists a FAISS index to a file.
    
    Args:
        index (faiss.Index): The FAISS index to save.
        path (str): The file path to save the index.
    """
    try:
        faiss.write_index(index, path)
        logger.info(f"Successfully persisted index to {path}.")
    except Exception as e:
        logger.error(f"Failed to persist index to {path}: {e}")
        raise