# cv_factory/shared_libs/feature_store/base/base_retriever.py (Updated)

import abc
from typing import Any, List, Dict, Optional

class BaseRetriever(abc.ABC):
    """
    Abstract Base Class for retrieval mechanisms.

    This interface defines a standard way to retrieve relevant information, 
    often incorporating filtering and re-ranking logic.
    """

    @abc.abstractmethod
    def retrieve(self, query: Any, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant items based on a query, with optional metadata filtering.

        Args:
            query (Any): The input query (e.g., a query embedding vector or raw text).
            top_k (int): The number of results to retrieve.
            filters (Optional[Dict[str, Any]]): Metadata constraints (e.g., {"source": "S3"}).

        Returns:
            List[Dict[str, Any]]: A list of retrieved results.
        """
        raise NotImplementedError