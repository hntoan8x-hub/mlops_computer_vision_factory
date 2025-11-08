# shared_libs/data_processing/image_components/feature_extractors/image_feature_extractor_orchestrator.py

import logging
import numpy as np
import os
from typing import Dict, Any, List, Optional, Union

# --- Import Contracts and Configuration ---
from shared_libs.data_processing._base.base_feature_extractor import FeatureData
from shared_libs.data_processing.configs.preprocessing_config_schema import FeatureExtractionConfig

# --- Import Factory & Policy Controller (Self-contained Engine Logic) ---
from shared_libs.data_processing.image_components.feature_extractors.image_feature_extractor_factory import ImageFeatureExtractorFactory 
from shared_libs.data_processing.image_components.feature_extractors.feature_extractor_policy_controller import FeatureExtractorPolicyController 

logger = logging.getLogger(__name__)

# Define the input type for consistency
ImageData = Union[np.ndarray, List[np.ndarray]]

class ImageFeatureExtractorOrchestrator:
    """
    Orchestrates and executes a sequence of feature extraction and embedding steps (Feature Engineering Pipeline).

    This class acts as the decoupled execution engine, integrating the Policy Controller for 
    conditional execution based on input metadata.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating configuration, building components, 
        and initializing the Policy Controller.

        Args:
            config (Dict[str, Any]): A dictionary containing the feature extraction configuration.

        Raises:
            ValueError: If the configuration fails Pydantic validation.
        """
        try:
            # 1. Configuration Validation (Quality Gate)
            self.config_schema: FeatureExtractionConfig = FeatureExtractionConfig(**config)
        except Exception as e:
            logger.error(f"FeatureExtractionConfig Validation Failed. Error: {e}")
            raise ValueError(f"Invalid FeatureExtraction Configuration provided: {e}")
            
        # 2. Hardening: Initialize Policy Controller
        # Read policy_mode from the configuration dictionary
        self._policy_controller = FeatureExtractorPolicyController(
            policy_type=config.get('policy_mode', 'default') 
        )
            
        # 3. Pipeline Construction (Build ALL enabled components)
        self._all_components: List[Any] = [] # List of BaseFeatureExtractor/BaseEmbedder instances
        for step in self.config_schema.components: 
            if step.enabled:
                component = ImageFeatureExtractorFactory.create(
                    component_type=step.type,
                    config=step.params
                )
                self._all_components.append(component)

        logger.info(f"ImageFeatureExtractorOrchestrator initialized with {len(self._all_components)} components. Policy: {self._policy_controller.policy_type}")

    def transform(self, data: ImageData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> FeatureData:
        """
        Sequentially applies feature extraction/embedding selected by the Policy Controller.

        Args:
            data (ImageData): Preprocessed image data.
            metadata (Optional[Dict[str, Any]]): Metadata (e.g., resolution, data_type) 
                                                 used by the policy controller for conditional execution.
            **kwargs: Additional keyword arguments passed to underlying components.

        Returns:
            FeatureData: The final extracted feature vector(s).
        """
        if not self.config_schema.enabled or not self._all_components:
            logger.debug("Feature extraction is disabled or pipeline is empty. Skipping.")
            return data # Note: Returns ImageData if pipeline is empty/disabled. Check final expected output type.

        # 1. CRITICAL: Policy selects the actual pipeline to run (Conditional Execution)
        selected_pipeline = self._policy_controller.select_pipeline(
            self._all_components, 
            metadata=metadata
        )
        
        current_data = data
        for component in selected_pipeline:
            # Execution logic: Call .extract() or .embed() defined in the Base classes.
            if hasattr(component, 'extract'):
                current_data = component.extract(current_data, **kwargs)
            elif hasattr(component, 'embed'):
                current_data = component.embed(current_data, **kwargs)
            else:
                 logger.warning(f"Component {component.__class__.__name__} does not implement required method. Skipping.")
            
            logger.debug(f"Executed feature component: {component.__class__.__name__}")
            
        logger.debug("Feature extraction pipeline completed.")
        return current_data

    # --- MLOps Lifecycle Methods (Delegating State Management) ---

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'ImageFeatureExtractorOrchestrator':
        """
        Fits stateful components in the pipeline (e.g., Dim Reducer, Feature Scaler).

        Args:
            X (ImageData): Input image data used for fitting.
            y (Optional[Any]): Target data (optional).

        Returns:
            ImageFeatureExtractorOrchestrator: The fitted orchestrator instance.
        """
        for step in self._all_components:
            if hasattr(step, 'fit'):
                step.fit(X, y) 
        logger.info("Feature extraction pipeline fitting completed.")
        return self

    def save(self, directory_path: str) -> None:
        """
        Saves the state of the entire feature pipeline by delegating saving to each component.

        Args:
            directory_path (str): The file path where the pipeline state will be saved.
        """
        os.makedirs(directory_path, exist_ok=True)
        for i, step in enumerate(self._all_components):
            try:
                # File name includes index and class name for safety
                file_name = f"{step.__class__.__name__}_{i}.pkl" 
                if hasattr(step, 'save'):
                    step.save(os.path.join(directory_path, file_name))
            except Exception as e:
                logger.error(f"Error saving state for {step.__class__.__name__}: {e}")
        logger.info(f"Feature extraction pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Loads the state of the entire feature pipeline, re-initializing components based on the config.

        Args:
            directory_path (str): The file path from which the pipeline state will be loaded.
        """
        # Reconstruct the pipeline based on current config first (CRITICAL for consistency)
        self._all_components = []
        for i, step_config in enumerate(self.config_schema.components):
            if step_config.enabled:
                component = ImageFeatureExtractorFactory.create(
                    component_type=step_config.type, 
                    config=step_config.params
                )
                
                file_name = f"{component.__class__.__name__}_{i}.pkl"
                file_path = os.path.join(directory_path, file_name)
                
                # Only load if the file exists and the component supports loading state
                if os.path.exists(file_path) and hasattr(component, 'load'):
                    component.load(file_path)
                
                self._all_components.append(component)
                
        logger.info(f"Feature extraction pipeline state loaded from {directory_path}.")