# shared_libs/data_processing/image_components/augmenters/image_augmenter_orchestrator.py

import logging
import os
import pickle # Used for save/load of individual augmenters
from typing import Dict, Any, Optional, List, Union

# --- Import Contracts and Configuration --
from shared_libs.data_processing._base.base_augmenter import BaseAugmenter, ImageData 
from shared_libs.data_processing.configs.preprocessing_config_schema import AugmentationConfig 

# --- CRITICAL NEW IMPORTS ---
from shared_libs.data_processing.image_components.augmenters.policy_controller import AugmentPolicyController 
from shared_libs.data_processing.image_components.augmenters.image_augmenter_factory import ImageAugmenterFactory 

logger = logging.getLogger(__name__)

class ImageAugmenterOrchestrator:
    """
    Orchestrates and executes a sequence of data augmentation steps, supporting policy-based selection (RandAugment).
    
    This class manages configuration validation and acts as its own execution engine, 
    ensuring decoupling from the ml_core layer.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating configuration, building atomic components, 
        and initializing the Policy Controller.

        Args:
            config (Dict[str, Any]): A dictionary containing the augmentation configuration.
        """
        try:
            # 1. Configuration Validation (Quality Gate)
            self.config_schema: AugmentationConfig = AugmentationConfig(**config)
        except Exception as e:
            logger.error(f"AugmentationConfig Validation Failed. Error: {e}")
            raise ValueError(f"Invalid Augmentation Configuration provided: {e}")
            
        # 2. Hardening: Initialize Policy Controller based on config schema
        self._policy_controller = AugmentPolicyController(
            policy_type=self.config_schema.policy_mode,
            n_select=self.config_schema.n_select,
            # Pass a default magnitude (e.g., 0.5) if the config doesn't specify one 
            # (assuming magnitude can be passed via extra params or defaults)
            magnitude=config.get('magnitude', 0.5) 
        )
            
        # 3. Build Atomic Components (Full Pipeline)
        self._all_augmenters: List[BaseAugmenter] = []
        for step in self.config_schema.steps:
            if step.enabled:
                component = ImageAugmenterFactory.create(
                    augmenter_type=step.type,
                    config=step.params
                )
                self._all_augmenters.append(component)

        logger.info(f"AugmenterOrchestrator initialized with {len(self._all_augmenters)} atomic augmenters.")


    def transform(self, data: ImageData, **kwargs: Dict[str, Any]) -> ImageData:
        """
        Applies augmentation based on the configured policy, passing the determined magnitude 
        to the selected components.

        Args:
            data (ImageData): The input image(s) to be augmented.
            **kwargs: Additional keyword arguments (e.g., 'labels' for CutMix/Mixup).

        Returns:
            ImageData: The augmented image(s).
        """
        if not self.config_schema.enabled or not self._all_augmenters:
            logger.debug("Augmentation is disabled or pipeline is empty. Skipping.")
            return data

        # 1. CRITICAL: Policy selects the pipeline AND the magnitude (M)
        selected_pipeline, M = self._policy_controller.select_pipeline(self._all_augmenters)
        
        processed_data = data
        
        # 2. Sequential execution of the selected steps
        for step in selected_pipeline:
            # CRITICAL CHANGE: Pass the determined magnitude (M) to the atomic component's transform method
            processed_data = step.transform(processed_data, magnitude=M, **kwargs)
            logger.debug(f"Executed augmenter: {step.__class__.__name__} with M={M:.2f}")
            
        logger.debug("Policy-based data augmentation transformation completed.")
        return processed_data

    # --- MLOps Lifecycle Delegation ---

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'ImageAugmenterOrchestrator':
        """Fits any stateful components (though augmenters are typically stateless)."""
        for step in self._all_augmenters:
            if hasattr(step, 'fit'):
                step.fit(X, y)
        logger.info("Augmentation pipeline fitting completed.")
        return self

    def save(self, directory_path: str) -> None:
        """Saves the state of the entire augmentation pipeline (parameters of all atomic components)."""
        os.makedirs(directory_path, exist_ok=True)
        for i, step in enumerate(self._all_augmenters):
            try:
                file_name = f"{step.__class__.__name__}_{i}.pkl" 
                if hasattr(step, 'save'):
                    step.save(os.path.join(directory_path, file_name))
            except Exception as e:
                logger.error(f"Error saving state for {step.__class__.__name__}: {e}")
        logger.info(f"Augmentation pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """Loads the state of the entire augmentation pipeline."""
        self._all_augmenters = []
        for i, step_config in enumerate(self.config_schema.steps):
            if step_config.enabled:
                component = ImageAugmenterFactory.create(
                    augmenter_type=step_config.type, 
                    config=step_config.params
                )
                file_name = f"{component.__class__.__name__}_{i}.pkl"
                file_path = os.path.join(directory_path, file_name)
                
                if os.path.exists(file_path) and hasattr(component, 'load'):
                    component.load(file_path)
                
                self._all_augmenters.append(component)
                
        logger.info(f"Augmentation pipeline state loaded from {directory_path}.")