# cv_factory/shared_libs/feature_store/factories/vector_store_factory.py

import logging
from typing import Dict, Any, Type, Optional
from shared_libs.feature_store.base.base_vector_store import BaseVectorStore

# Import Concrete Connectors
from shared_libs.feature_store.connectors.pinecone_connector import PineconeConnector
from shared_libs.feature_store.connectors.milvus_connector import MilvusConnector
from shared_libs.feature_store.connectors.faiss_connector import FaissConnector
from shared_libs.feature_store.connectors.chromadb_connector import ChromaDBConnector
from shared_libs.feature_store.connectors.weaviate_connector import WeaviateConnector

# Import Utilities (ConfigManager for Configuration Loading)
from shared_libs.core_utils.config_manager import ConfigManager 
from shared_libs.core_utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """
    Factory for creating concrete instances of BaseVectorStore connectors.
    
    This centralizes the instantiation and ensures the correct parameters 
    are passed to distributed/managed vector databases.
    """

    _CONNECTOR_MAP: Dict[str, Type[BaseVectorStore]] = {
        "pinecone": PineconeConnector,
        "milvus": MilvusConnector,
        "faiss": FaissConnector,
        "chromadb": ChromaDBConnector,
        "weaviate": WeaviateConnector,
    }

    @staticmethod
    def create(connector_type: str, config: Dict[str, Any]) -> BaseVectorStore:
        """
        Creates and returns a concrete vector store connector instance.

        Args:
            connector_type (str): The type of vector store (e.g., 'pinecone', 'faiss').
            config (Dict[str, Any]): Validated configuration parameters for the connector.

        Returns:
            BaseVectorStore: An instance of the requested connector.

        Raises:
            ConfigurationError: If the connector type is unsupported or creation fails.
        """
        connector_type = connector_type.lower()
        ConnectorClass = VectorStoreFactory._CONNECTOR_MAP.get(connector_type)
        
        if not ConnectorClass:
            supported_types = ", ".join(VectorStoreFactory._CONNECTOR_MAP.keys())
            raise ConfigurationError(f"Unsupported vector store type: '{connector_type}'. Supported: {supported_types}")

        try:
            # Instantiate the connector using the configuration dictionary
            return ConnectorClass(**config)
        except Exception as e:
            logger.critical(f"Failed to instantiate {connector_type} connector: {e}")
            raise ConfigurationError(f"Connector creation failed for '{connector_type}': {e}")