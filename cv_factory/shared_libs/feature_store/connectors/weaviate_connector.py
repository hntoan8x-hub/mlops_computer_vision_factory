# cv_factory/shared_libs/feature_store/connectors/weaviate_connector.py

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from uuid import uuid4

import weaviate
from weaviate.data.replication import ConsistencyLevel

from shared_libs.feature_store.base.base_vector_store import BaseVectorStore # The updated Base

logger = logging.getLogger(__name__)

class WeaviateConnector(BaseVectorStore):
    """
    A concrete implementation of BaseVectorStore for Weaviate.

    This connector interacts with a Weaviate cluster for storing and searching vectors,
    implementing all required data management and operational methods.
    """
    # Standard class name for all feature vectors
    CLASS_NAME = "ImageEmbedding"

    def __init__(self, host: str = "http://localhost", port: int = 8080, api_key: Optional[str] = None):
        """
        Initializes parameters but defers connection until connect() is called.
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.client: Optional[weaviate.Client] = None
        self.is_connected = False
        self.class_name = self.CLASS_NAME

    # --- Connection Management ---
    
    def connect(self) -> None:
        """
        Establishes the Weaviate client connection and ensures the schema is set up.
        """
        if self.is_connected:
            return

        try:
            # Weaviate client initialization
            self.client = weaviate.Client(
                url=f"{self.host}:{self.port}",
                additional_headers={"X-OpenAI-Api-Key": self.api_key} if self.api_key else None
            )
            
            # Ensure the required class (schema) exists
            self._ensure_class_exists()
            self.is_connected = True
            logger.info(f"Connected to Weaviate at {self.host}:{self.port}. Class: {self.class_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise ConnectionError(f"Weaviate connection failed: {e}")
            
    def _ensure_class_exists(self) -> None:
        """Ensures the vector class exists in the Weaviate schema."""
        if not self.client:
            raise RuntimeError("Weaviate client is not initialized. Call connect() first.")
            
        try:
            schema = self.client.schema.get()
            class_exists = any(c['class'] == self.class_name for c in schema.get('classes', []))
            
            if not class_exists:
                class_obj = {
                    "class": self.class_name,
                    "properties": [
                        {
                            "dataType": ["text"],
                            "name": "vector_id", # Property to store the generated ID as metadata
                        }
                    ],
                    "vectorIndexType": "hnsw" # Default index type
                }
                self.client.schema.create_class(class_obj)
                logger.info(f"Created new Weaviate class '{self.class_name}'.")

        except Exception as e:
            logger.error(f"Failed to manage Weaviate schema: {e}")
            raise

    def close(self) -> None:
        """
        Closes the connector. As a persistent database, this is mostly a logging step.
        """
        # Weaviate client manages its connections, so we just reset the state.
        self.is_connected = False
        self.client = None
        logger.info("Weaviate connector closed.")

    # --- Data Management (C-U-D) ---

    def add_embeddings(self, embeddings: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Adds a batch of embeddings and returns generated IDs.
        """
        if not self.client: self.connect()
        
        ids: List[str] = []
        
        # Weaviate performs better with batching
        with self.client.batch(consistency_level=ConsistencyLevel.ALL) as batch:
            for i, vec in enumerate(embeddings):
                vec_id = str(uuid4())
                meta = metadata[i] if metadata and len(metadata) > i else {}
                meta['vector_id'] = vec_id # Store the vector_id as a property
                
                batch.add_data_object(
                    data_object=meta,
                    class_name=self.class_name,
                    vector=vec.tolist(),
                    uuid=vec_id # Use the same UUID for the object ID
                )
                ids.append(vec_id)
                
        logger.info(f"Added {len(ids)} embeddings to Weaviate.")
        return ids

    def delete_embeddings(self, item_ids: List[str]) -> None:
        """
        Deletes embeddings from the index based on their unique IDs.
        """
        if not self.client: self.connect()

        # Delete operations are typically batched for efficiency
        with self.client.batch(consistency_level=ConsistencyLevel.ALL) as batch:
            for item_id in item_ids:
                batch.delete(
                    uuid=item_id,
                    class_name=self.class_name,
                )
        logger.info(f"Deleted {len(item_ids)} embeddings from Weaviate.")

    def update_metadata(self, item_id: str, new_metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata associated with a specific embedding ID.
        """
        if not self.client: self.connect()
        
        # Weaviate's update operation is a PATCH (partial update)
        self.client.data_object.update(
            data_object=new_metadata,
            class_name=self.class_name,
            uuid=item_id,
            consistency_level=ConsistencyLevel.ALL
        )
        logger.info(f"Updated metadata for Weaviate ID: {item_id}.")

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Retrieves the raw embedding vector for a given ID.
        """
        if not self.client: self.connect()
        
        data_object = self.client.data_object.get_by_id(
            uuid=item_id,
            class_name=self.class_name,
            with_vector=True # Crucial flag to include the vector data
        )
        
        if data_object and 'vector' in data_object:
            # Vector is returned as a list, convert to numpy array
            return np.array(data_object['vector'])
        return None

    # --- Retrieval (Enhanced with Filters) ---

    def search(self, query_vector: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search, now supporting metadata filters.
        """
        if not self.client: self.connect()
        
        query_builder = self.client.query.get(
            self.class_name, ["vector_id", "_additional {certainty distance}"] 
            # Request vector_id and similarity metadata (certainty/distance)
        ).with_near_vector({
            "vector": query_vector.tolist()
        }).with_limit(top_k)

        # Apply metadata filtering if provided
        if filters:
            # NOTE: Filters must be transformed into Weaviate's 'where' filter structure.
            # Assuming a simple key-value matching for this implementation.
            # Example: filters = {"label": "cat"} -> where_filter = {"path": ["label"], "operator": "Equal", "valueText": "cat"}
            
            # This is a placeholder for a robust filter translation logic.
            where_filter = self._translate_filters_to_weaviate(filters)
            query_builder = query_builder.with_where(where_filter)
        
        result = query_builder.do()
        
        retrieved_results = []
        if not result.get('data') or not result['data'].get('Get'):
            return []
            
        for item in result['data']['Get'][self.class_name]:
            additional = item.pop('_additional', {})
            distance = additional.get('distance')
            
            retrieved_results.append({
                "id": item.get('vector_id', item.get('uuid')),
                "distance": distance,
                "metadata": {k: v for k, v in item.items() if k != "vector_id"}
            })
        return retrieved_results
    
    def _translate_filters_to_weaviate(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates a simple key-value filter dict into Weaviate's 'where' clause.
        A more robust implementation would handle complex operators (GT, LT, AND, OR).
        """
        if not filters:
            return {}
        
        # Simple AND logic for all key-value pairs
        where_filter = {"operator": "And", "operands": []}
        
        for key, value in filters.items():
            filter_operand = {
                "path": [key],
                "operator": "Equal"
            }
            if isinstance(value, str):
                filter_operand["valueText"] = value
            elif isinstance(value, int):
                filter_operand["valueInt"] = value
            elif isinstance(value, float):
                filter_operand["valueNumber"] = value
            elif isinstance(value, bool):
                filter_operand["valueBoolean"] = value
            else:
                logger.warning(f"Unsupported filter type for key '{key}': {type(value)}")
                continue
                
            where_filter["operands"].append(filter_operand)

        return where_filter

    # --- State and Persistence (Persistent Database Specific) ---

    def persist(self, index_path: str) -> None:
        """
        Weaviate is a persistent database; no manual 'persist' method is generally needed. 
        Operation is a No-Op.
        """
        logger.info("Weaviate is a persistent database; persist operation is a No-Op.")

    def get_state(self) -> Dict[str, Any]:
        """Returns the necessary parameters for re-connecting."""
        return {
            "host": self.host, 
            "port": self.port, 
            "api_key": self.api_key
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restores parameters. Requires re-calling connect() to establish connection."""
        self.__init__(**state)
        self.connect()