# cv_factory/shared_libs/feature_store/connectors/chromadb_connector.py

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import QueryResult # For type hinting results

from shared_libs.feature_store.base.base_vector_store import BaseVectorStore # The updated Base

logger = logging.getLogger(__name__)

class ChromaDBConnector(BaseVectorStore):
    """
    A concrete implementation of BaseVectorStore for ChromaDB.

    This connector uses a local/persistent ChromaDB instance to store and search vector embeddings,
    implementing all required data management and operational methods.
    """
    
    def __init__(self, collection_name: str, path: str = "./chroma_db", **kwargs):
        """
        Initializes parameters but defers connection until connect() is called.
        """
        self.collection_name = collection_name
        self.db_path = path
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
    # --- NEW: Connection Management ---

    def connect(self) -> None:
        """
        Establishes the connection to the persistent ChromaDB client and collection.
        """
        if self.client is None:
            # Use PersistentClient for durability in production environments
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Note: Model_name=None is used if embeddings are pre-calculated externally
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=None)
            )
            logger.info(f"Connected to ChromaDB collection '{self.collection_name}' at '{self.db_path}'.")

    def close(self) -> None:
        """
        Closes the connection to the vector store (persists data).
        """
        if self.client:
            # Persistent clients handle persistence implicitly or via persist() call, 
            # but we ensure logging and resource awareness.
            self.client.persist()
            logger.info("ChromaDB connector closed and persisted.")

    # --- Data Management (C-U-D) ---

    def add_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Adds a batch of embeddings and their associated metadata. Returns generated IDs.
        """
        if not self.collection: self.connect()
        
        ids = [str(uuid4()) for _ in range(embeddings.shape[0])]
        metadatas = metadata if metadata else [{}] * embeddings.shape[0]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {embeddings.shape[0]} embeddings to ChromaDB.")
        return ids

    def delete_embeddings(self, item_ids: List[str]) -> None:
        """
        Deletes embeddings from the store based on their unique IDs (required for GDPR/Data Hygiene).
        """
        if not self.collection: self.connect()
        self.collection.delete(ids=item_ids)
        logger.info(f"Deleted {len(item_ids)} embeddings from ChromaDB.")

    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata associated with a specific embedding ID.
        """
        if not self.collection: self.connect()
        # ChromaDB uses 'update' to modify metadata/documents/embeddings based on ID
        self.collection.update(
            ids=[item_id],
            metadatas=[new_metadata]
        )
        logger.info(f"Updated metadata for ID: {item_id}.")

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieves the raw embedding vector for a given ID (for validation or debugging).
        """
        if not self.collection: self.connect()
        results = self.collection.get(
            ids=[item_id],
            include=['embeddings']
        )
        
        if results and results.get('embeddings'):
            return np.array(results['embeddings'][0])
        return None

    # --- Retrieval ---

    def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search, now supporting metadata filters.
        """
        if not self.collection: self.connect()
        
        # ChromaDB uses 'where' clause for metadata filtering
        results: QueryResult = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=top_k,
            where=filters if filters else None, # Integrate filters
            include=['metadatas', 'distances']
        )
        
        retrieved_results = []
        if 'ids' in results and results['ids']:
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i],
                    "metadata": results['metadatas'][0][i],
                    # Add raw embedding vector if needed, but it's often omitted for speed
                }
                retrieved_results.append(result)
        return retrieved_results

    # --- State and Persistence ---

    def persist(self, index_path: str) -> None:
        """Saves the vector store's index to disk (explicitly calls client.persist())."""
        if self.client:
            self.client.persist()
            logger.info("ChromaDB explicitly persisted.")
            
    def get_state(self) -> Dict[str, Any]:
        """State management is handled by persistence in ChromaDB."""
        return {"collection_name": self.collection_name, "db_path": self.db_path, "status": "persisted"}

    def set_state(self, state: Dict[str, Any]) -> None:
        """State management is handled by persistence in ChromaDB."""
        logger.warning("ChromaDB set_state typically relies on re-initializing the PersistentClient with the saved path.")
        if state.get('db_path'):
             self.__init__(state['collection_name'], state['db_path'])