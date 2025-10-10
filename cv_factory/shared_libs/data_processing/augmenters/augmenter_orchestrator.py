# cv_factory/shared_libs/data_processing/augmenters/augmenter_orchestrator.py (RESTRUCTURED)

import logging
from typing import Dict, Any, Optional

# --- Import Contracts and Execution Engine ---
# NOTE: BaseAugmenter and ImageData are likely in shared_libs.data_processing._base
from shared_libs.data_processing._base.base_augmenter import ImageData 
from shared_libs.data_processing.configs.preprocessing_config_schema import AugmentationConfig # Config Schema for validation
from shared_libs.ml_core.pipeline_components_cv.orchestrator.component_orchestrator import ComponentOrchestrator 

logger = logging.getLogger(__name__)

class AugmenterOrchestrator:
    """
    Orchestrates a sequence of data augmentation steps.
    
    This class acts as a FAÇADE, delegating the pipeline lifecycle (fit, transform, save, load) 
    to the ComponentOrchestrator engine via Composition.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating the configuration and building the pipeline engine.

        Args:
            config (Dict[str, Any]): A dictionary containing the augmentation configuration.
        """
        # 1. Configuration Validation: Ensure the input config is valid
        self.config_schema = AugmentationConfig(**config)
        
        # 2. Prepare Pipeline Configuration for the Execution Engine
        pipeline_config = [step.dict() for step in self.config_schema.steps]

        # 3. Initialize the Execution Engine (Composition)
        # This engine manages the sequence of BaseAugmenter components
        self._pipeline_engine: ComponentOrchestrator = ComponentOrchestrator(
            config=pipeline_config
        )
        
        logger.info("AugmenterOrchestrator initialized. Pipeline execution delegated.")

    # NOTE: The redundant _build_pipeline() method is REMOVED.

    # --- Core Pipeline Delegation Methods ---

    def transform(self, data: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Delegates the sequential transformation (augmentation) to the ComponentOrchestrator.
        
        This method must accept **kwargs (e.g., labels) to support advanced augmentation 
        techniques like CutMix or Mixup.

        Args:
            data (ImageData): The input image(s) to be augmented.
            **kwargs: Additional keyword arguments passed to underlying components.

        Returns:
            ImageData: The augmented image(s).
        """
        # Delegation: The core transformation step, passing along any extra args
        processed_data = self._pipeline_engine.transform(data, **kwargs)
        logger.debug("Data augmentation transformation completed.")
        return processed_data

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'AugmenterOrchestrator':
        """
        Delegates the fitting process to the ComponentOrchestrator.
        
        Though augmenters are typically stateless, this is included for architectural consistency.
        """
        self._pipeline_engine.fit(X, y)
        logger.info("Augmentation pipeline fitting completed.")
        return self

    # --- MLOps Lifecycle Delegation ---

    def save(self, directory_path: str) -> None:
        """
        Delegates saving the state of the entire augmentation pipeline.
        """
        self._pipeline_engine.save(directory_path)
        logger.info(f"Augmentation pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Delegates loading the state of the entire augmentation pipeline.
        """
        self._pipeline_engine.load(directory_path)
        logger.info(f"Augmentation pipeline state loaded from {directory_path}.")