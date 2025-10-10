import abc
from typing import List, Dict, Any, Union, Optional
import numpy as np

class BaseVectorStore(abc.ABC):
    """
    Abstract Base Class for vector storage backends.

    This interface defines the essential operations for any vector database
    or indexing mechanism, ensuring a consistent API for storing, searching,
    and managing embeddings.
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """
        Establishes the connection (network or local) to the vector store index.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Adds a batch of embeddings and their associated metadata to the store.

        Args:
            embeddings (np.ndarray): A 2D NumPy array where each row is an embedding vector.
            metadata (Optional[List[Dict[str, Any]]]): A list of dictionaries, where each
                                                      dictionary corresponds to the metadata
                                                      of an embedding.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search for the nearest neighbors of a query vector.

        Args:
            query_vector (np.ndarray): The query embedding vector.
            top_k (int): The number of nearest neighbors to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of results, each containing information about
                                  the retrieved embedding (e.g., ID, distance, metadata).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete_embeddings(self, item_ids: List[str]) -> None:
        """
        Deletes embeddings from the store based on their unique IDs.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata associated with a specific embedding ID.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieves the raw embedding vector for a given ID (for validation or debugging).
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def persist(self, index_path: str) -> None:
        """
        Saves the vector store's index to a specified path.
        This is crucial for local backends like FAISS.

        Args:
            index_path (str): The path to save the index file.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the vector store.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Sets the state of the vector store from a previously saved state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Closes the connection to the vector store.
        """
        raise NotImplementedError