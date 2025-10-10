# cv_factory/shared_libs/feature_store/orchestrator/feature_store_orchestrator.py

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

# Import Abstractions and Factories
from shared_libs.feature_store.base.base_retriever import BaseRetriever
from shared_libs.feature_store.base.base_vector_store import BaseVectorStore
from shared_libs.feature_store.factories.vector_store_factory import VectorStoreFactory
from shared_libs.feature_store.factories.retriever_factory import RetrieverFactory
from shared_libs.ml_core.configs.feature_store_config_schema import FeatureStoreConfig # Assuming config schema is in ml_core/configs

logger = logging.getLogger(__name__)

class FeatureStoreOrchestrator:
    """
    Orchestrates the entire feature store workflow, managing the lifecycle (connect/close), 
    data indexing (CRUD), and complex retrieval strategies.

    This class serves as the main entry point, integrating the Vector Store (I/O) 
    and the Retriever (Logic).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by building the vector store and retriever based on validated config.
        """
        # CRITICAL: Validate config against the hardened schema
        self.config = FeatureStoreConfig(**config) 
        self.vector_store: Optional[BaseVectorStore] = None
        self.retriever: Optional[BaseRetriever] = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes the vector store and retriever based on the parsed configuration.
        """
        # Step 1: Initialize the Vector Store (Connector)
        try:
            vector_store_config = self.config.vector_store
            self.vector_store = VectorStoreFactory.create(
                connector_type=vector_store_config.type,
                config=vector_store_config.connection_params # Pass connection params directly
            )
            # CRITICAL: Connect is deferred until first use or explicitly called by lifecycle manager
        except Exception as e:
            logger.critical(f"Failed to initialize vector store connector: {e}")
            raise

        # Step 2: Initialize the Retriever (Logic)
        try:
            retriever_config = self.config.retriever
            # Dependency Injection: Pass the vector store instance to the retriever
            dependencies = {"vector_store": self.vector_store}
            self.retriever = RetrieverFactory.create(
                retriever_type=retriever_config.type,
                dependencies=dependencies,
                config=retriever_config.params or {} # Ensure params is a dict
            )
        except Exception as e:
            logger.critical(f"Failed to initialize retriever logic: {e}")
            raise

    # --- Lifecycle Management (Context Manager/Explicit Calls) ---

    def __enter__(self):
        """Connects the vector store upon entering the context."""
        if self.vector_store:
            self.vector_store.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the vector store connection upon exiting the context."""
        self.close()
        # Propagate exception if needed
        return False
        
    def close(self) -> None:
        """Closes the connection to the feature store backend."""
        if self.vector_store:
            self.vector_store.close()
            logger.info("Feature store orchestrator closed connections.")
    
    # --- Data Indexing and CRUD (Create, Read, Update, Delete) ---

    def index_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Indexes a batch of embeddings and returns the generated IDs."""
        if self.vector_store is None: raise RuntimeError("Vector store is not initialized.")
        # Ensure connection is established before operation
        if not self.vector_store.is_connected: self.vector_store.connect() 
        
        ids = self.vector_store.add_embeddings(embeddings, metadata)
        logger.info(f"Successfully indexed {embeddings.shape[0]} embeddings.")
        return ids

    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """Updates the metadata associated with a specific embedding ID."""
        if self.vector_store is None: raise RuntimeError("Vector store is not initialized.")
        if not self.vector_store.is_connected: self.vector_store.connect() 

        self.vector_store.update_metadata(item_id, new_metadata)
        logger.info(f"Metadata updated for ID: {item_id}.")

    def delete_embeddings(self, item_ids: List[str]) -> None:
        """Deletes embeddings from the store based on their unique IDs (GDPR compliance)."""
        if self.vector_store is None: raise RuntimeError("Vector store is not initialized.")
        if not self.vector_store.is_connected: self.vector_store.connect() 

        self.vector_store.delete_embeddings(item_ids)
        logger.info(f"Deleted {len(item_ids)} embeddings.")

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Retrieves the raw embedding vector for a given ID (for validation or debugging)."""
        if self.vector_store is None: raise RuntimeError("Vector store is not initialized.")
        if not self.vector_store.is_connected: self.vector_store.connect() 
        
        return self.vector_store.get_embedding(item_id)
        
    # --- Retrieval ---
        
    def search_embeddings(self, query: Any, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Searches for the most similar embeddings using the configured retriever, supporting filters.

        Args:
            query (Any): The input query (e.g., query embedding vector or text dict for hybrid).
            top_k (int): The number of results to retrieve.
            filters (Optional[Dict[str, Any]]): Metadata constraints to filter the search results.

        Returns:
            List[Dict[str, Any]]: A list of retrieved results, sorted by relevance.
        """
        if self.retriever is None: raise RuntimeError("Retriever is not initialized.")
        # Retriever internally calls vector_store.search, so we rely on the vector store 
        # being connected, which is handled by the context manager.
        
        results = self.retriever.retrieve(query, top_k, filters)
        logger.info(f"Search completed. Found {len(results)} results (Filters: {bool(filters)}).")
        return results

    def persist(self, index_path: Optional[str] = None) -> None:
        """
        Persists the vector store index to a file (used for local backends like FAISS).
        """
        if self.vector_store is None: raise RuntimeError("Vector store is not initialized.")
        
        # Use a default path from config if not provided
        if not index_path:
            index_path = self.config.vector_store.connection_params.get("index_path")
        
        if index_path:
            self.vector_store.persist(index_path)
            logger.info(f"Vector store index explicitly persisted to {index_path}.")
        else:
            logger.warning("No index path provided. Skipping persistence (This is normal for managed services).")