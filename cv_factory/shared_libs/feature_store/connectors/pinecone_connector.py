# cv_factory/shared_libs/feature_store/connectors/pinecone_connector.py

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from uuid import uuid4

# NOTE: Using the standard pinecone imports
from pinecone import Pinecone, Index

from shared_libs.feature_store.base.base_vector_store import BaseVectorStore # The updated Base

logger = logging.getLogger(__name__)

class PineconeConnector(BaseVectorStore):
    """
    A concrete implementation of BaseVectorStore for Pinecone.

    This connector interacts with a Pinecone index for storing and searching vectors,
    implementing all required data management and operational methods for a managed service.
    """
    def __init__(self, api_key: str, index_name: str, environment: str, dim: int, metric: str = "cosine"):
        """
        Initializes parameters but defers connection until connect() is called.
        """
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.dim = dim
        self.metric = metric
        self.pinecone_client: Optional[Pinecone] = None
        self.index: Optional[Index] = None
        self.is_connected = False

    # --- Connection Management (Updated to fit BaseVectorStore) ---
    
    def connect(self) -> None:
        """
        Establishes the Pinecone client connection and connects to the index.
        """
        if self.is_connected:
            return

        try:
            self.pinecone_client = Pinecone(api_key=self.api_key, environment=self.environment)
            
            # Check for index existence and create if necessary (This logic is for initialization, 
            # ideally the index is provisioned via Terraform/IaC)
            if self.index_name not in self.pinecone_client.list_indexes():
                self.pinecone_client.create_index(self.index_name, dimension=self.dim, metric=self.metric)
                logger.info(f"Created new Pinecone index '{self.index_name}'.")
                
            self.index = self.pinecone_client.Index(self.index_name)
            self.is_connected = True
            logger.info(f"Connected to Pinecone index '{self.index_name}'.")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise ConnectionError(f"Pinecone connection failed: {e}")

    def close(self) -> None:
        """
        Closes the connector. As a managed service, this is mostly a logging step.
        """
        self.is_connected = False
        logger.info("Pinecone connector closed.")

    # --- Data Management (C-U-D) ---

    def add_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Adds a batch of embeddings via upsert and returns generated IDs.
        """
        if not self.index: self.connect()
        
        vectors = []
        ids: List[str] = []
        
        for i, vec in enumerate(embeddings):
            vec_id = str(uuid4())
            meta = metadata[i] if metadata and len(metadata) > i else {}
            vectors.append((vec_id, vec.tolist(), meta))
            ids.append(vec_id)
            
        # Pinecone uses 'upsert' for adding/updating vectors
        self.index.upsert(vectors=vectors)
        logger.info(f"Added {len(vectors)} embeddings to Pinecone.")
        return ids

    def delete_embeddings(self, item_ids: List[str]) -> None:
        """
        Deletes embeddings from the index based on their unique IDs.
        """
        if not self.index: self.connect()
        self.index.delete(ids=item_ids)
        logger.info(f"Deleted {len(item_ids)} embeddings from Pinecone.")

    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata associated with a specific embedding ID.
        """
        if not self.index: self.connect()
        
        # Pinecone uses the upsert method with an existing ID to update metadata
        # NOTE: This requires retrieving the original vector first if we only want to update metadata, 
        # but standard practice is to use the dedicated update endpoint or metadata_update feature.
        
        # Simplified update (assuming the vector is not modified)
        self.index.update(
            id=item_id,
            set_metadata=new_metadata
        )
        logger.info(f"Updated metadata for Pinecone ID: {item_id}.")

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieves the raw embedding vector for a given ID.
        """
        if not self.index: self.connect()
        
        # Fetch the vector data
        results = self.index.fetch(ids=[item_id])
        vector = results.get('vectors', {}).get(item_id, {}).get('values')
        
        if vector:
            return np.array(vector)
        return None

    # --- Retrieval ---

    def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search, now supporting metadata filters.
        """
        if not self.index: self.connect()
        
        query_results = self.index.query(
            vector=query_vector.tolist(), 
            top_k=top_k, 
            include_metadata=True,
            filter=filters if filters else None # Integrate metadata filtering
        )
        
        results = []
        for match in query_results['matches']:
            results.append({
                "id": match['id'],
                "distance": match['score'],
                "metadata": match.get('metadata', {})
            })
        return results

    # --- State and Persistence (Managed Service Specific) ---

    def persist(self, index_path: str) -> None:
        """
        Pinecone is a managed service; no manual 'persist' method is needed. 
        We enforce this contract by logging the No-Op.
        """
        logger.info("Pinecone is a managed service; persist operation is a No-Op.")

    def get_state(self) -> Dict[str, Any]:
        """Returns the necessary parameters for re-connecting."""
        return {
            "index_name": self.index_name, 
            "environment": self.environment, 
            "dim": self.dim
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores parameters. Requires re-calling connect() to establish connection."""
        self.__init__(self.api_key, **state)
        self.connect()