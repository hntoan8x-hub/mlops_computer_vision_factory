# cv_factory/shared_libs/feature_store/factories/retriever_factory.py

import logging
from typing import Dict, Any, Type, Optional, List
from shared_libs.feature_store.base.base_retriever import BaseRetriever
from shared_libs.feature_store.base.base_vector_store import BaseVectorStore

# Import Concrete Retrievers
from shared_libs.feature_store.retrievers.dense_retriever import DenseRetriever
from shared_libs.feature_store.retrievers.hybrid_retriever import HybridRetriever
from shared_libs.feature_store.retrievers.reranker import Reranker

# Import Utilities
from shared_libs.core_utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """
    Factory for creating concrete instances of BaseRetriever (Logic).

    This factory handles dependency injection of the VectorStore into the retriever logic.
    """

    _RETRIEVER_MAP: Dict[str, Type[BaseRetriever]] = {
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
        "reranker": Reranker,
    }

    @staticmethod
    def create(retriever_type: str, dependencies: Dict[str, Any], config: Dict[str, Any]) -> BaseRetriever:
        """
        Creates and returns a concrete retriever instance.

        Args:
            retriever_type (str): The type of retrieval strategy (e.g., 'dense', 'hybrid').
            dependencies (Dict[str, Any]): Mandatory dependencies like {'vector_store': BaseVectorStore_instance}.
            config (Dict[str, Any]): Validated configuration parameters for the retriever.

        Returns:
            BaseRetriever: An instance of the requested retriever.

        Raises:
            ConfigurationError: If the retriever type is unsupported or creation fails.
        """
        retriever_type = retriever_type.lower()
        RetrieverClass = RetrieverFactory._RETRIEVER_MAP.get(retriever_type)

        if not RetrieverClass:
            supported_types = ", ".join(RetrieverFactory._RETRIEVER_MAP.keys())
            raise ConfigurationError(f"Unsupported retriever type: '{retriever_type}'. Supported: {supported_types}")

        # CRITICAL: Validate required dependencies (Dependency Injection Check)
        if retriever_type != "reranker" and "vector_store" not in dependencies:
            raise ConfigurationError(f"Retriever type '{retriever_type}' requires 'vector_store' dependency.")
        
        try:
            # Handle specific dependencies based on the class constructor signature
            if retriever_type == "hybrid":
                # NOTE: HybridRetriever requires both Dense and Sparse retrievers as dependencies
                # This requires a higher-level orchestrator to pass them or to recursively call the factory.
                return RetrieverClass(
                    dense_retriever=dependencies.get("dense_retriever"),
                    sparse_retriever=dependencies.get("sparse_retriever"),
                    **config
                )
            
            # Default case (Dense/Reranker): passes vector_store directly
            return RetrieverClass(
                vector_store=dependencies.get("vector_store"),
                **config
            )
        except Exception as e:
            logger.critical(f"Failed to instantiate {retriever_type} retriever: {e}")
            raise ConfigurationError(f"Retriever creation failed for '{retriever_type}': {e}")