# cv_factory/shared_libs/data_processing/feature_extractors/feature_extractor_orchestrator.py (PRODUCTION RESTRUCTURED)

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

# --- Import Contracts and Execution Engine ---
# NOTE: FeatureData and BaseFeatureExtractor are likely in _base/
from shared_libs.data_processing._base.base_feature_extractor import FeatureData
from shared_libs.data_processing.configs.preprocessing_config_schema import FeatureExtractionConfig
from shared_libs.ml_core.pipeline_components_cv.orchestrator.component_orchestrator import ComponentOrchestrator 

# Define the input type for consistency
ImageData = Union[np.ndarray, List[np.ndarray]]

logger = logging.getLogger(__name__)

class FeatureExtractorOrchestrator:
    """
    Orchestrates a sequence of classical feature extraction steps.
    
    This class acts as a FAÇADE, delegating the pipeline lifecycle (fit, transform/extract, save, load) 
    to the ComponentOrchestrator engine via Composition.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating the configuration and building the pipeline engine.

        Args:
            config (Dict[str, Any]): A dictionary containing the feature extraction configuration.
        """
        # 1. Configuration Validation: Enforce schema validation
        self.config_schema = FeatureExtractionConfig(**config)
        
        # 2. Prepare Pipeline Configuration for the Execution Engine
        pipeline_config = [step.dict() for step in self.config_schema.steps]

        # 3. Initialize the Execution Engine (Composition)
        # This engine manages the sequence of BaseFeatureExtractor components
        self._pipeline_engine: ComponentOrchestrator = ComponentOrchestrator(
            config=pipeline_config
        )
        
        logger.info("FeatureExtractorOrchestrator initialized. Pipeline execution delegated.")

    # NOTE: The redundant _build_pipeline() method is REMOVED.

    # --- Core Pipeline Delegation Methods ---
    
    # We use 'extract' as the public method name but delegate to the engine's 'transform'
    def extract(self, data: ImageData) -> List[FeatureData]:
        """
        Delegates the sequential feature extraction process to the ComponentOrchestrator.
        
        Args:
            data (ImageData): The input image(s).

        Returns:
            List[FeatureData]: The list of extracted features.
        """
        # Delegation: The core transformation step
        extracted_features = self._pipeline_engine.transform(data)
        logger.debug("Feature extraction process completed.")
        return extracted_features

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'FeatureExtractorOrchestrator':
        """
        Delegates the fitting process to the ComponentOrchestrator.
        
        This is crucial if any feature extractor component (e.g., dictionary learning, feature scaling) is stateful.
        """
        self._pipeline_engine.fit(X, y)
        logger.info("Feature extraction pipeline fitting completed.")
        return self

    # --- MLOps Lifecycle Delegation ---

    def save(self, directory_path: str) -> None:
        """
        Delegates saving the state of the entire feature extraction pipeline.
        """
        self._pipeline_engine.save(directory_path)
        logger.info(f"Feature extraction pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Delegates loading the state of the entire feature extraction pipeline.
        """
        self._pipeline_engine.load(directory_path)
        logger.info(f"Feature extraction pipeline state loaded from {directory_path}.")