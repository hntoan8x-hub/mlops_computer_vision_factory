# cv_factory/shared_libs/data_processing/embedders/embedder_orchestrator.py (PRODUCTION RESTRUCTURED)

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

# --- Import Contracts and Execution Engine ---
from shared_libs.data_processing._base.base_embedder import EmbeddingData 
from shared_libs.data_processing.configs.preprocessing_config_schema import FeatureExtractionConfig # Using FeatureExtractionConfig for embedder validation
from shared_libs.ml_core.pipeline_components_cv.orchestrator.component_orchestrator import ComponentOrchestrator 

# Define the input type for consistency with other orchestrators
ImageData = Union[np.ndarray, List[np.ndarray]]

logger = logging.getLogger(__name__)

class EmbedderOrchestrator:
    """
    Orchestrates a sequence of deep learning embedding steps.
    
    This class acts as a FAÇADE, delegating the pipeline lifecycle (fit, transform/embed, save, load) 
    to the ComponentOrchestrator engine via Composition.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating the configuration and building the pipeline engine.

        Args:
            config (Dict[str, Any]): A dictionary containing the embedding configuration.
        """
        # 1. Configuration Validation: Enforce schema validation
        self.config_schema = FeatureExtractionConfig(**config)
        
        # 2. Prepare Pipeline Configuration for the Execution Engine
        pipeline_config = [step.dict() for step in self.config_schema.steps]

        # 3. Initialize the Execution Engine (Composition)
        self._pipeline_engine: ComponentOrchestrator = ComponentOrchestrator(
            config=pipeline_config
        )
        
        logger.info("EmbedderOrchestrator initialized. Pipeline execution delegated.")

    # NOTE: The redundant _build_pipeline() method is REMOVED.

    # --- Core Pipeline Delegation Methods ---
    
    # We use 'embed' as the public method name but delegate to the engine's 'transform'
    def embed(self, data: ImageData) -> List[EmbeddingData]:
        """
        Delegates the sequential embedding process to the ComponentOrchestrator.
        
        Args:
            data (ImageData): The input image(s).

        Returns:
            List[EmbeddingData]: The list of extracted embeddings.
        """
        # Delegation: The core transformation step
        # Note: If embedding yields a list of outputs, it relies on ComponentOrchestrator's 
        # ability to handle sequential transformation on the entire batch/list structure.
        extracted_embeddings = self._pipeline_engine.transform(data)
        logger.debug("Deep learning embedding process completed.")
        return extracted_embeddings

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'EmbedderOrchestrator':
        """
        Delegates the fitting process to the ComponentOrchestrator.
        
        This is crucial if the embedding process includes a stateful layer (e.g., BatchNorm frozen statistics).
        """
        self._pipeline_engine.fit(X, y)
        logger.info("Embedding pipeline fitting completed.")
        return self

    # --- MLOps Lifecycle Delegation ---

    def save(self, directory_path: str) -> None:
        """
        Delegates saving the state of the entire embedding pipeline.
        """
        self._pipeline_engine.save(directory_path)
        logger.info(f"Embedding pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Delegates loading the state of the entire embedding pipeline.
        """
        self._pipeline_engine.load(directory_path)
        logger.info(f"Embedding pipeline state loaded from {directory_path}.")