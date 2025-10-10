# cv_factory/shared_libs/feature_store/retrievers/reranker.py

import logging
from typing import List, Dict, Any, Union, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pickle

from shared_libs.feature_store.base.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class Reranker(BaseRetriever):
    """
    A stateful retriever component that re-ranks initial search results using a 
    contextual cross-encoder model (e.g., from Hugging Face).

    It implements the BaseRetriever contract and manages the state of the 
    large Transformer model and its tokenizer.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Reranker with a cross-encoder model.

        Args:
            model_name (str): The name of the pre-trained cross-encoder model from Hugging Face.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Load model and tokenizer upon initialization (or use connect/load)
        self._load_model_and_tokenizer()

        logger.info(f"Initialized Reranker with model '{model_name}' on device '{self.device}'.")

    def _load_model_and_tokenizer(self):
        """Internal method to load the cross-encoder model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load Reranker model/tokenizer for {self.model_name}: {e}")
            raise RuntimeError(f"Reranker initialization failed.")

    # --- Implement BaseRetriever Contract ---
    
    # NOTE: The Reranker's standard input is (query_text, initial_results). 
    # We enforce the BaseRetriever signature and assume 'query' is a dict containing 'text'.
    def retrieve(self, query: Dict[str, Any], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs the re-ranking operation on a list of initial search results.
        
        It assumes that the 'query' dictionary contains a key 'initial_results' 
        which holds the list of items to be re-ranked (from a Dense/Hybrid search).
        
        Args:
            query (Dict[str, Any]): Must contain the 'text' query and typically 'initial_results'.
            top_k (int): The final number of results to return after re-ranking.
            filters (Optional[Dict[str, Any]]): Filters are IGNORED here, as filtering 
                                                should occur in the initial retrieval step.

        Returns:
            List[Dict[str, Any]]: The re-ranked list of results, sorted by relevance score.
        """
        initial_results: List[Dict[str, Any]] = query.get("initial_results", [])
        query_text = query.get("text")
        
        if not initial_results or not query_text:
            logger.warning("Reranker received empty input or missing query text. Returning empty list.")
            return []

        # 1. Prepare input pairs for the cross-encoder
        # Each result must contain 'metadata' with a 'text' field (the document chunk)
        sentences = [[query_text, res['metadata'].get('text', '')] for res in initial_results]
        
        try:
            # 2. Tokenization and Inference
            with torch.no_grad():
                features = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
                scores = self.model(**features).logits
                
                # Squeeze score dimension and convert to NumPy
                scores = scores.squeeze(1).cpu().numpy()
                
            # 3. Apply scores and Re-rank
            for i, score in enumerate(scores):
                initial_results[i]['rerank_score'] = score.item()
            
            initial_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranking completed. Returned top {top_k} results.")
            return initial_results[:top_k]
        
        except Exception as e:
            logger.error(f"Failed to perform reranking operation: {e}")
            return initial_results[:top_k] # Return the initial top results as a fallback

    # --- Stateful Management (Required by BaseComponent/MLOps) ---
    
    def save(self, path: str) -> None:
        """
        Saves the state of the reranker model (weights and tokenizer config).
        """
        # Save model weights
        model_path = os.path.join(path, "model.pt")
        tokenizer_path = os.path.join(path, "tokenizer")
        
        # Save model state dictionary
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer configuration (crucial for Hugging Face models)
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save metadata about the model name
        with open(os.path.join(path, "metadata.pkl"), 'wb') as f:
            pickle.dump({'model_name': self.model_name}, f)
            
        logger.info(f"Reranker state (model weights and tokenizer) saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the reranker model state from the specified path.
        """
        # 1. Load model metadata and config
        with open(os.path.join(path, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            
        self.model_name = metadata['model_name']
        model_path = os.path.join(path, "model.pt")
        tokenizer_path = os.path.join(path, "tokenizer")
        
        # 2. Re-instantiate based on saved config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 3. Load model weights
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device).eval()
        
        logger.info(f"Reranker model loaded from {path} and placed on {self.device}.")
        
    # NOTE: Since Reranker is not part of the standard BaseVectorStore hierarchy, 
    # the connect/close methods are not strictly enforced here, but the save/load 
    # methods are critical for its lifecycle management as a stateful component.