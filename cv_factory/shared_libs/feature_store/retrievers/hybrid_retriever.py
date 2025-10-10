# cv_factory/shared_libs/feature_store/retrievers/hybrid_retriever.py

import logging
import numpy as np
from typing import List, Dict, Any, Union
from collections import defaultdict

from shared_libs.feature_store.base.base_retriever import BaseRetriever
# from shared_libs.feature_store.base.base_vector_store import BaseVectorStore # Not strictly needed here, but good context

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    A retriever that combines results from a dense search with a sparse search
    (e.g., BM25) and re-ranks them based on a fusion algorithm.
    
    This implementation uses a simple reciprocal rank fusion (RRF) for simplicity,
    which is robust as it only requires the *rank* of the results, not the scores
    to be comparable.
    """
    def __init__(self, dense_retriever: BaseRetriever, sparse_retriever: BaseRetriever, **kwargs: Dict[str, Any]):
        """
        Initializes the HybridRetriever.

        Args:
            dense_retriever (BaseRetriever): A dense retriever instance (e.g., DenseRetriever).
            sparse_retriever (BaseRetriever): A sparse retriever instance (e.g., SparseRetriever/BM25Retriever).
            **kwargs: Additional parameters for fusion (e.g., k for RRF, which is often a constant).
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        # RRF constant k: typically set to a small value (e.g., 60) for smoothing.
        self.rrf_k = kwargs.get("rrf_k", 60) 
        logger.info(f"Initialized HybridRetriever with RRF constant k={self.rrf_k}.")

    def _fused_score(self, rank: int) -> float:
        """Calculates the Reciprocal Rank Fusion score for a given rank."""
        return 1.0 / (self.rrf_k + rank)

    def _fuse_results(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuses and re-ranks the results using Reciprocal Rank Fusion (RRF).
        
        RRF Formula: score_new = sum( 1 / (k + rank_i) )
        """
        fused_scores = defaultdict(float)
        all_results_map = {} # To store the latest/most complete metadata for each document ID

        # 1. Fuse Dense Results
        for rank, res in enumerate(dense_results, 1):
            doc_id = res['id']
            # Compute RRF score contribution
            fused_scores[doc_id] += self._fused_score(rank)
            # Store metadata
            all_results_map[doc_id] = res

        # 2. Fuse Sparse Results
        for rank, res in enumerate(sparse_results, 1):
            doc_id = res['id']
            # Compute RRF score contribution
            fused_scores[doc_id] += self._fused_score(rank)
            # Store metadata (prioritize metadata from the second result set if ID is new)
            if doc_id not in all_results_map:
                all_results_map[doc_id] = res

        # 3. Create a final list and sort
        final_results = []
        for doc_id, score in fused_scores.items():
            result = all_results_map[doc_id].copy() # Start with one of the source results
            result['fused_score'] = score
            final_results.append(result)
            
        # Sort by the new fused score in descending order
        final_results.sort(key=lambda x: x['fused_score'], reverse=True)
        
        return final_results

    def retrieve(self, query: Dict[str, Union[np.ndarray, str]], top_k: int) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search by combining dense and sparse results.

        The 'query' dictionary must contain:
        - 'dense_vector': The query embedding vector (for DenseRetriever).
        - 'text': The raw text query (for SparseRetriever).

        Args:
            query (Dict[str, Union[np.ndarray, str]]): The query dictionary with 
                'dense_vector' (np.ndarray/List[float]) and 'text' (str) keys.
            top_k (int): The number of results to retrieve after fusion.

        Returns:
            List[Dict[str, Any]]: A list of fused and re-ranked results.
        """
        if 'dense_vector' not in query or 'text' not in query:
            raise ValueError("Hybrid retriever requires 'dense_vector' (for dense search) and 'text' (for sparse search) in the query dictionary.")
        
        # 1. Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query['dense_vector'], top_k)
        sparse_results = self.sparse_retriever.retrieve(query['text'], top_k)

        # 2. Fuse and re-rank the results
        fused_results = self._fuse_results(dense_results, sparse_results)

        logger.info(f"Hybrid retriever found and fused {len(fused_results)} unique results, returning top {top_k}.")
        
        # 3. Return the top_k results
        return fused_results[:top_k]