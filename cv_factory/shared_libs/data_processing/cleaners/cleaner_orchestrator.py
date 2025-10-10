# cv_factory/shared_libs/data_processing/cleaners/cleaner_orchestrator.py (PRODUCTION RESTRUCTURED)

import logging
from typing import Dict, Any, Optional

# --- Import Contracts and Execution Engine ---
# NOTE: Assuming ImageData is defined in base_image_cleaner
from shared_libs.data_processing._base.base_image_cleaner import ImageData 
from shared_libs.data_processing.configs.preprocessing_config_schema import CleaningConfig
from shared_libs.ml_core.pipeline_components_cv.orchestrator.component_orchestrator import ComponentOrchestrator 

logger = logging.getLogger(__name__)

class CleanerOrchestrator:
    """
    Orchestrates a sequence of image cleaning steps.
    
    This class acts as a FAÇADE, using the ComponentOrchestrator via Composition 
    to manage the actual pipeline execution lifecycle (fit, transform, save, load).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by delegating pipeline construction to ComponentOrchestrator.

        Args:
            config (Dict[str, Any]): A dictionary containing the cleaning configuration.
        """
        # 1. Configuration Validation: Enforce schema validation (Quality Gate)
        self.config_schema = CleaningConfig(**config)
        
        # 2. Prepare Pipeline Configuration for the Execution Engine
        # ComponentOrchestrator expects a List[Dict], not Pydantic objects
        pipeline_config = [step.dict() for step in self.config_schema.steps]

        # 3. Initialize the Execution Engine (Composition)
        # This engine handles the actual creation and sequential running of BaseImageCleaner components
        self._pipeline_engine: ComponentOrchestrator = ComponentOrchestrator(
            config=pipeline_config
        )
        
        logger.info("CleanerOrchestrator initialized. Pipeline execution delegated.")

    # --- Core Pipeline Delegation Methods ---

    def transform(self, data: ImageData) -> ImageData:
        """
        Delegates the sequential transformation (cleaning) to the ComponentOrchestrator.
        
        Args:
            data (ImageData): The input image(s) to be cleaned.

        Returns:
            ImageData: The cleaned image(s).
        """
        # Delegation: The core transformation step
        processed_data = self._pipeline_engine.transform(data)
        logger.debug("Image cleaning transformation completed.")
        return processed_data

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'CleanerOrchestrator':
        """
        Delegates the fitting process to the ComponentOrchestrator.
        
        This is necessary if any cleaning component (e.g., PCA, robust scaler) is stateful.
        """
        # Delegation: Fit any stateful components
        self._pipeline_engine.fit(X, y)
        logger.info("Cleaning pipeline fitting completed.")
        return self

    # --- MLOps Lifecycle Delegation ---

    def save(self, directory_path: str) -> None:
        """
        Delegates saving the state of the entire cleaning pipeline (including fitted parameters).
        """
        # Delegation: Save the engine's state
        self._pipeline_engine.save(directory_path)
        logger.info(f"Cleaning pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Delegates loading the state of the entire cleaning pipeline, ensuring consistency across environments.
        """
        # Delegation: Load the engine's state
        self._pipeline_engine.load(directory_path)
        logger.info(f"Cleaning pipeline state loaded from {directory_path}.")