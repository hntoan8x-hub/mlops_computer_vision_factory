# shared_libs/data_processing/image_components/cleaners/image_cleaner_orchestrator.py

import logging
import os
import pickle
from typing import Dict, Any, Optional, List

# --- Import Contracts and Configuration ---
from shared_libs.data_processing._base.base_image_cleaner import BaseImageCleaner, ImageData 
from shared_libs.data_processing.configs.preprocessing_config_schema import CleaningConfig

# --- Import Factory & Policy Controller (Self-contained Engine Logic) ---
from shared_libs.data_processing.image_components.cleaners.image_cleaner_factory import ImageCleanerFactory 
from shared_libs.data_processing.image_components.cleaners.cleaner_policy_controller import CleanerPolicyController 

logger = logging.getLogger(__name__)

class ImageCleanerOrchestrator:
    """
    Orchestrates and executes a sequence of image cleaning steps (Image Cleaning Pipeline).
    
    This class manages configuration validation, acts as its own execution engine 
    (decoupled from ml_core), and integrates the CleanerPolicyController for 
    conditional/adaptive cleaning based on metadata.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating configuration, building atomic components, 
        and initializing the Policy Controller.

        Args:
            config (Dict[str, Any]): A dictionary containing the cleaning configuration.
        """
        try:
            # 1. Configuration Validation (Quality Gate)
            self.config_schema: CleaningConfig = CleaningConfig(**config)
        except Exception as e:
            logger.error(f"CleaningConfig Validation Failed. Error: {e}")
            raise ValueError(f"Invalid Cleaning Configuration provided: {e}")
            
        # 2. Hardening: Initialize Policy Controller
        # Read policy_mode from the configuration dictionary (assuming it's passed at this level)
        self._policy_controller = CleanerPolicyController(
            policy_type=config.get('policy_mode', 'default') 
        )
            
        # 3. Pipeline Construction (Self-contained Execution Engine)
        self._all_cleaners: List[BaseImageCleaner] = []
        for step in self.config_schema.steps:
            if step.enabled:
                component = ImageCleanerFactory.create(
                    cleaner_type=step.type,
                    config=step.params
                )
                self._all_cleaners.append(component)

        logger.info(f"ImageCleanerOrchestrator initialized with {len(self._all_cleaners)} active steps. Policy: {self._policy_controller.policy_type}")

    # --- Core Pipeline Execution Methods ---

    def transform(self, data: ImageData, metadata: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Sequentially applies the transformation (cleaning) of components selected by the Policy Controller.
        
        Args:
            data (ImageData): The input image(s) to be cleaned.
            metadata (Optional[Dict[str, Any]]): Metadata (e.g., color format) used by the policy controller 
                                                 for conditional execution.
            **kwargs: Additional keyword arguments.

        Returns:
            ImageData: The cleaned image(s).
        """
        if not self.config_schema.enabled or not self._all_cleaners:
            logger.debug("Cleaning is disabled or pipeline is empty. Skipping.")
            return data

        # 1. CRITICAL: Policy selects the actual pipeline to run (Conditional Execution)
        selected_pipeline = self._policy_controller.select_and_configure_pipeline(
            self._all_cleaners, 
            metadata=metadata
        )
        
        current_data = data
        for component in selected_pipeline:
            # Core execution logic: pass data through the selected pipeline
            current_data = component.transform(current_data, metadata=metadata, **kwargs)
        
        logger.debug("Image cleaning transformation completed.")
        return current_data

    # --- MLOps Lifecycle Methods (Simplified Save/Load) ---

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'ImageCleanerOrchestrator':
        """
        Fits any stateful components in the pipeline (e.g., if a component like PCA or Scaler were used).

        Args:
            X (ImageData): The training image data.
            y (Optional[Any]): Target labels.

        Returns:
            ImageCleanerOrchestrator: The fitted orchestrator instance.
        """
        for step in self._all_cleaners:
            if hasattr(step, 'fit'):
                step.fit(X, y)
        logger.info("Cleaning pipeline fitting completed.")
        return self

    def save(self, directory_path: str) -> None:
        """
        Saves the state of the entire cleaning pipeline (parameters of all atomic components).

        Args:
            directory_path (str): The file path where the pipeline state will be saved.
        """
        os.makedirs(directory_path, exist_ok=True)
        for i, step in enumerate(self._all_cleaners):
            try:
                # File name includes index for reconstruction safety
                file_name = f"{step.__class__.__name__}_{i}.pkl" 
                if hasattr(step, 'save'):
                    step.save(os.path.join(directory_path, file_name))
            except Exception as e:
                logger.error(f"Error saving state for {step.__class__.__name__}: {e}")
        logger.info(f"Cleaning pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Loads the state of the entire cleaning pipeline.

        Args:
            directory_path (str): The file path from which the pipeline state will be loaded.
        """
        # Reconstruct the pipeline based on the current config first
        self._all_cleaners = []
        for i, step_config in enumerate(self.config_schema.steps):
            if step_config.enabled:
                component = ImageCleanerFactory.create(
                    cleaner_type=step_config.type, 
                    config=step_config.params
                )
                file_name = f"{component.__class__.__name__}_{i}.pkl"
                file_path = os.path.join(directory_path, file_name)
                
                if os.path.exists(file_path) and hasattr(component, 'load'):
                    component.load(file_path)
                
                self._all_cleaners.append(component)
                
        logger.info(f"Cleaning pipeline state loaded from {directory_path}.")