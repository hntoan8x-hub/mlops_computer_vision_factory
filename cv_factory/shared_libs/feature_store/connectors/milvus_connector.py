# cv_factory/shared_libs/feature_store/connectors/milvus_connector.py

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType, MilvusException

from shared_libs.feature_store.base.base_vector_store import BaseVectorStore # The updated Base

logger = logging.getLogger(__name__)

class MilvusConnector(BaseVectorStore):
    """
    A concrete implementation of BaseVectorStore for Milvus.

    This connector interacts with a Milvus cluster to store and search vector embeddings,
    implementing all required data management and operational methods for a distributed system.
    """
    
    def __init__(self, alias: str, collection_name: str, dim: int, host: str = "localhost", port: int = 19530):
        """
        Initializes parameters and defers actual connection until connect() is called.
        """
        self.alias = alias
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.collection: Optional[Collection] = None
        self.is_connected = False
        
        # --- NEW: Primary Key Field Name (Milvus requirement) ---
        self.pk_field = "pk"
        self.vector_field = "embedding"
        
        # NOTE: Collection is created on connect()

    # --- Connection Management ---
    
    def connect(self) -> None:
        """
        Establishes the connection to the Milvus cluster and loads the collection.
        """
        if self.is_connected:
            return

        try:
            if not connections.has_connection(self.alias):
                connections.connect(self.alias, host=self.host, port=self.port)
                logger.info(f"Connected to Milvus at {self.host}:{self.port} with alias '{self.alias}'.")
            
            self._create_collection_if_not_exists()
            self.is_connected = True
            
        except MilvusException as e:
            logger.error(f"Milvus connection/creation failed: {e}")
            raise ConnectionError(f"Milvus connection failed: {e}")

    def _create_collection_if_not_exists(self) -> None:
        """Internal helper to ensure the collection exists and is indexed."""
        if utility.has_collection(self.collection_name, using=self.alias):
            self.collection = Collection(self.collection_name, using=self.alias)
        else:
            fields = [
                FieldSchema(name=self.pk_field, dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                # NOTE: In production, metadata fields should be explicitly defined here for indexing/filtering
            ]
            schema = CollectionSchema(fields, f"Milvus collection for {self.collection_name}")
            self.collection = Collection(self.collection_name, schema, using=self.alias)

        # Ensure index exists and collection is loaded into memory
        if not self.collection.has_index():
            index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            self.collection.create_index(field_name=self.vector_field, index_params=index_params)
            
        self.collection.load()
        logger.info(f"Milvus collection '{self.collection_name}' ready.")

    # --- Data Management (C-U-D) ---
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Adds a batch of embeddings. Returns the generated/used primary keys (IDs).
        """
        if not self.collection: self.connect()
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}.")
        
        insert_data = []
        ids: List[str] = []
        
        for i, vec in enumerate(embeddings):
            pk = str(uuid4()) # Use UUID as primary key
            ids.append(pk)
            
            # Milvus requires a list of columns for insertion
            entity = {self.pk_field: pk, self.vector_field: vec.tolist()}
            
            # Merge external metadata (note: only defined fields will be inserted)
            if metadata and len(metadata) > i:
                entity.update(metadata[i])
            insert_data.append(entity)
        
        # Prepare data for PyMilvus insert: list of dictionaries
        self.collection.insert(insert_data)
        self.collection.flush() # Ensure data is written to disk segments
        
        logger.info(f"Added {len(ids)} embeddings to Milvus collection '{self.collection_name}'.")
        return ids

    def delete_embeddings(self, item_ids: List[str]) -> None:
        """
        Deletes embeddings from the collection based on their primary keys.
        """
        if not self.collection: self.connect()
        
        # Milvus uses expression-based deletion
        expr = f"{self.pk_field} in {item_ids}"
        self.collection.delete(expr)
        self.collection.flush()
        
        logger.info(f"Deleted {len(item_ids)} embeddings from Milvus.")

    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata associated with a specific embedding ID.
        
        NOTE: Milvus does not support direct update of vector or primary key. 
        It supports Upsert for metadata/scalar fields if the collection is defined correctly.
        For simplicity, we rely on the client to handle the upsert or deletion/re-insertion.
        """
        logger.warning("Milvus update strategy is complex. Currently relies on external logic or Milvus Upsert capability for metadata.")
        # Actual implementation requires a careful upsert or delete-and-reinsert operation.
        
        # Example of re-inserting (non-optimal, but guaranteed to update metadata)
        # self.collection.delete(f"pk == '{item_id}'")
        # self.collection.insert(...)


    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieves the raw embedding vector for a given primary key ID.
        """
        if not self.collection: self.connect()
        
        # Milvus uses query() to retrieve entities
        expr = f"{self.pk_field} == '{item_id}'"
        results = self.collection.query(
            expr=expr,
            output_fields=[self.vector_field],
            limit=1
        )
        
        if results and self.vector_field in results[0]:
            return np.array(results[0][self.vector_field])
        return None

    # --- Retrieval ---

    def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search, now supporting metadata filters (Milvus 'expr').
        """
        if not self.collection: self.connect()
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Milvus filtering: convert dictionary filters to Milvus expression string if needed
        filter_expr = "" 
        if filters:
            # Example: {"label": "cat"} -> "label == 'cat'"
            logger.warning("Milvus filter conversion not fully implemented. Assuming caller provides Milvus expression.")
            # filter_expr = self._convert_filters_to_milvus_expr(filters) 
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=self.vector_field,
            param=search_params,
            limit=top_k,
            expr=filter_expr, # Integrate filters via expression
            output_fields=[self.pk_field] # Retrieve primary keys
        )
        
        retrieved_results = []
        for hit in results[0]:
            retrieved_results.append({
                "id": str(hit.id), # Ensure ID is string
                "distance": hit.distance,
            })
        return retrieved_results

    # --- State and Persistence ---

    def persist(self, index_path: str) -> None:
        """
        Flushes the collection segments to disk for durability (Milvus's persistence mechanism).
        """
        if self.collection:
            self.collection.flush()
            logger.info(f"Milvus collection '{self.collection_name}' flushed.")

    def get_state(self) -> Dict[str, Any]:
        """State management is handled by the distributed nature of Milvus."""
        logger.warning("Returning basic connection state. True state management requires complex distributed queries.")
        return {"alias": self.alias, "collection_name": self.collection_name, "is_distributed": True}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Milvus relies on connection parameters, not state restoration via dictionary."""
        logger.warning("Milvus does not restore state via dictionary; relies on connection. Use connect().")
    
    def close(self) -> None:
        """Disconnects the client from the Milvus cluster."""
        if connections.has_connection(self.alias):
            connections.disconnect(self.alias)
            logger.info(f"Milvus connection '{self.alias}' disconnected.")