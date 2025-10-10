# cv_factory/shared_libs/feature_store/retrievers/dense_retriever.py

import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional

from shared_libs.feature_store.base.base_retriever import BaseRetriever # The updated Base
from shared_libs.feature_store.base.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

class DenseRetriever(BaseRetriever):
    """
    A retriever that performs a similarity search on a vector store using a dense query vector.
    
    This retriever directly uses the search functionality of the underlying vector store, 
    including support for metadata filtering.
    """
    def __init__(self, vector_store: BaseVectorStore, **kwargs: Dict[str, Any]):
        """
        Initializes the retriever with a vector store instance.

        Args:
            vector_store (BaseVectorStore): An instance of a vector store connector (e.g., FaissConnector).
            **kwargs: Additional parameters.
        """
        if not isinstance(vector_store, BaseVectorStore):
            raise TypeError("vector_store must be an instance of BaseVectorStore.")
        self.vector_store = vector_store
        logger.info("Initialized DenseRetriever.")

    # --- Implement BaseRetriever Contract ---
    
    def retrieve(self, query: Union[np.ndarray, List[float]], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a dense search to retrieve the top_k most similar vectors,
        applying optional metadata filters via the vector store.

        Args:
            query (Union[np.ndarray, List[float]]): The query embedding vector.
            top_k (int): The number of results to retrieve.
            filters (Optional[Dict[str, Any]]): Metadata constraints to filter the search results.

        Returns:
            List[Dict[str, Any]]: A list of retrieved results, each containing ID, distance, and metadata.
        """
        if not isinstance(query, np.ndarray):
            query_vector = np.array(query, dtype=np.float32)
        else:
            query_vector = query
            
        # Ensure query is 2D for vector store search API compatibility
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        try:
            # CRITICAL: Pass the filters directly to the BaseVectorStore.search() method
            # The Vector Store Connector (e.g., Pinecone/Weaviate) is responsible for 
            # translating the filter dictionary into its native query language.
            results = self.vector_store.search(
                query_vector=query_vector, 
                top_k=top_k, 
                filters=filters
            )
            logger.info(f"Dense retriever found {len(results)} results (Filters applied: {bool(filters)}).")
            return results
        except Exception as e:
            logger.error(f"Failed to perform dense retrieval: {e}")
            raise