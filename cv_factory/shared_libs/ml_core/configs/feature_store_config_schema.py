# cv_factory/shared_libs/ml_core/configs/feature_store_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, PositiveInt, constr
from typing import List, Dict, Any, Union, Literal, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Assuming BaseConfig is available
class BaseConfig(BaseModel):
    class Config:
        extra = "allow" 
    enabled: bool = Field(True, description="Flag to enable/disable this component.")
    params: Optional[Dict[str, Any]] = Field(None, description="Component-specific parameters.")


# --- 1. Vector Store Connector Rules (Hardened) ---

class VectorStoreConfig(BaseConfig):
    """Rules for connecting to a specific Vector Database/Indexing Backend."""
    
    type: Literal["pinecone", "milvus", "faiss", "chromadb", "weaviate"] = Field(..., description="The Vector Store technology to use.")
    index_name: constr(min_length=3) = Field(..., description="Name of the index/collection to connect to.")
    
    # CRITICAL: Connection parameters are now mandatory
    connection_params: Dict[str, Any] = Field(..., description="Mandatory connection parameters (e.g., 'api_key', 'host', 'port', 'path').")
    
    embedding_dimension: PositiveInt = Field(..., description="Dimensionality of the vectors in the index (e.g., 512, 768).")
    
    @validator('connection_params')
    def check_mandatory_connection_keys(cls, v, values):
        """Rule: Enforce specific critical keys based on the connection type."""
        store_type = values.get('type')
        if store_type in ["pinecone", "milvus", "weaviate"] and ("host" not in v and "url" not in v):
             raise ValueError(f"{store_type} requires 'host' or 'url' connection parameter.")
        if store_type == "faiss" and "index_path" not in v and "dim" not in v:
             raise ValueError("FAISS requires 'dim' and preferred 'index_path' for persistence.")
        return v


# --- 2. Retriever Logic Rules (Hardened) ---

class RetrieverConfig(BaseConfig):
    """Rules for defining the retrieval strategy."""
    
    type: Literal["dense", "hybrid", "reranker"] = Field(..., description="The retrieval mechanism to use.")
    
    # Rules specific to Hybrid/Reranker logic
    rerank_model_name: Optional[str] = Field(None, description="Model ID for the re-ranking step (if type='reranker' or 'hybrid').")
    
    # Rule for Hybrid Search (using alpha for score fusion)
    alpha_weight: Field(default=0.5, ge=0.0, le=1.0) = Field(description="Weighting factor for dense vs. sparse components (only for type='hybrid').")
    
    @validator('rerank_model_name')
    def validate_reranker_if_needed(cls, v, values):
        """Rule: Reranker model is mandatory if reranker strategy is chosen."""
        if values.get('type') == "reranker" and not v:
            raise ValueError("Retriever type 'reranker' requires a 'rerank_model_name' to be provided.")
        return v


# --- 3. Master Feature Store Configuration ---

class FeatureStoreConfig(BaseConfig):
    """
    Master Schema for the Feature Store Orchestrator.
    This schema dictates the full setup, retrieval logic, and lifecycle management.
    """
    
    # Data Management
    feature_set_name: constr(min_length=3) = Field(..., description="The logical name of the feature set (e.g., 'CV_Image_Embeddings_V1').")
    
    # Connectors and Indexing
    vector_store: VectorStoreConfig = Field(..., description="Configuration for the underlying vector store.")
    
    # Retrieval Strategy
    retriever: RetrieverConfig = Field(..., description="Configuration for the retrieval logic layer.")
    
    # Lifecycle/Operational Parameters
    persistence_path: Optional[str] = Field(None, description="Default URI/Path for index persistence (used by local backends like FAISS).")
    
    # Global Search Parameters
    default_top_k: PositiveInt = Field(10, description="Default number of results to retrieve in a search operation.")
    
    @validator('vector_store')
    def validate_faiss_dim_match(cls, v):
        """Rule: Ensure FAISS dim matches the configured embedding dimension."""
        if v.type == "faiss" and v.embedding_dimension != v.connection_params.get("dim"):
            raise ValueError(
                "FAISS 'dim' in connection_params must explicitly match 'embedding_dimension' for consistency."
            )
        return v