# cv_factory/shared_libs/feature_store/connectors/faiss_connector.py

import faiss
import numpy as np
import logging
import os
from typing import List, Dict, Any, Union, Optional

from shared_libs.feature_store.base.base_vector_store import BaseVectorStore # The updated Base

logger = logging.getLogger(__name__)

class FaissConnector(BaseVectorStore):
    """
    A concrete implementation of BaseVectorStore for FAISS.

    It uses a local FAISS index for high-speed, in-memory similarity search and 
    manages metadata storage internally. Implements all required BaseVectorStore methods.
    """
    def __init__(self, dim: int, index_path: Optional[str] = None):
        """
        Initializes the parameters but defers actual index loading/creation until connect().
        """
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self.index_path = index_path
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.is_connected = False

    # --- NEW: Connection Management ---
    
    def connect(self) -> None:
        """
        Initializes or loads the FAISS index from disk.
        """
        if self.is_connected:
            return

        if self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # NOTE: Metadata must be loaded externally if not included in the index state
            logger.info(f"Loaded FAISS index from '{self.index_path}'.")
        else:
            # L2 distance is common for feature similarity
            self.index = faiss.IndexFlatL2(self.dim)
            logger.info(f"Initialized new FAISS index with dimension {self.dim}.")
            
        self.is_connected = True

    # --- Data Management (C-U-D) ---
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if not self.index: self.connect()
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}.")
        
        start_index = self.index.ntotal
        self.index.add(embeddings)
        
        ids: List[str] = []
        if metadata:
            if len(metadata) != embeddings.shape[0]:
                raise ValueError("Metadata list length must match the number of embeddings.")
            
            # FAISS uses internal sequential index IDs (integers)
            for i, meta in enumerate(metadata):
                internal_id = start_index + i
                self.metadata_store[internal_id] = meta
                # We return the internal FAISS index as the ID for tracking
                ids.append(str(internal_id)) 
        
        logger.info(f"Added {embeddings.shape[0]} embeddings to FAISS index.")
        return ids

    def delete_embeddings(self, item_ids: List[str]) -> None:
        """
        Deletes embeddings from the index based on internal IDs (FAISS requires a special index).
        
        NOTE: Simple IndexFlatL2 does not support deletion. This method should raise 
        an error or check if a removable index (e.g., IndexIDMap) is used. 
        For simplicity, we handle metadata deletion and log a warning.
        """
        if not self.index: self.connect()
        
        deleted_count = 0
        for item_id_str in item_ids:
            try:
                internal_id = int(item_id_str)
                if internal_id in self.metadata_store:
                    del self.metadata_store[internal_id]
                    deleted_count += 1
            except ValueError:
                logger.warning(f"Invalid ID format for deletion: {item_id_str}. Must be integer (FAISS internal ID).")

        # In a real system with a IndexIDMap or IndexFlat, we would call index.remove_ids([ids]) here.
        # Since we use IndexFlatL2, we rely on persistence/rebuild for physical deletion.
        logger.info(f"Deleted metadata for {deleted_count} embeddings. Physical index removal requires an IndexIDMap.")

    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata associated with a specific embedding ID.
        """
        try:
            internal_id = int(item_id)
            if internal_id in self.metadata_store:
                # Merge old metadata with new metadata
                self.metadata_store[internal_id].update(new_metadata)
                logger.info(f"Updated metadata for FAISS ID: {item_id}.")
            else:
                logger.warning(f"FAISS ID {item_id} not found in metadata store for update.")
        except ValueError:
            raise ValueError("Item ID must be an integer string representing the internal FAISS index.")

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieves the raw embedding vector for a given ID.
        """
        if not self.index: self.connect()
        try:
            internal_id = int(item_id)
            # FAISS uses read_vectors to retrieve the stored vector by its internal index ID
            vector = self.index.reconstruct(internal_id)
            return vector
        except Exception as e:
            logger.error(f"Error retrieving vector for ID {item_id}: {e}")
            return None

    # --- Retrieval ---

    def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search. FAISS IndexFlatL2 does not natively support filtering, 
        so filters are ignored here, but included for API compliance.
        """
        if not self.index: self.connect()
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        if filters:
             logger.warning("FAISS IndexFlatL2 does not support metadata filtering. Filters will be ignored.")

        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.metadata_store:
                result = self.metadata_store.get(idx, {}).copy()
                result['distance'] = dist
                result['id'] = str(idx) # Return ID as string for consistency with other connectors
                results.append(result)
        return results

    # --- State and Persistence ---
    
    def persist(self, index_path: str) -> None:
        """Saves the FAISS index to the specified path."""
        if self.index:
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS index persisted to '{index_path}'.")

    def get_state(self) -> Dict[str, Any]:
        """Returns the state needed for internal rebuild (metadata store)."""
        # Note: Index itself is saved via persist(). We save the dynamic metadata.
        return {"dim": self.dim, "metadata_store": self.metadata_store}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores the internal metadata store state."""
        self.dim = state["dim"]
        self.metadata_store = state["metadata_store"]
        logger.info("FAISS connector state (metadata) restored.")

    def close(self) -> None:
        """Closes the connector. Note: In-memory index is lost unless explicitly persisted."""
        self.is_connected = False
        logger.info("FAISS connector closed. In-memory index will be lost unless persisted.")