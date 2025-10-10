# cv_factory/shared_libs/ml_core/pipeline_components_cv/orchestrator/component_orchestrator.py

import logging
from typing import Dict, Any, List, Optional, Union
import os

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory

logger = logging.getLogger(__name__)

# Define the structure for a pipeline step configuration
PipelineConfig = List[Dict[str, Any]]

class ComponentOrchestrator:
    """
    The Execution Engine that orchestrates a sequence of CV pipeline components.

    This class builds a pipeline of BaseComponent Adapters based on a configuration 
    and applies them sequentially (in a Scikit-learn style).
    """

    def __init__(self, config: PipelineConfig, validator: Optional[Any] = None):
        """
        Initializes the orchestrator by building the pipeline.

        Args:
            config (PipelineConfig): An ordered list of dictionaries, where each 
                                     dictionary represents a pipeline step (Component).
            validator (Optional[Any]): An optional validator object (Placeholder for future use).
        """
        self.config = config
        self.pipeline: List[BaseComponent] = []
        self.validator = validator
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Builds the pipeline by creating component instances from the config using ComponentFactory.
        """
        for step_config in self.config:
            try:
                # Assuming the config has 'type' and 'params' fields
                component_type = step_config.get("type")
                component_params = step_config.get("params", {})
                
                # Delegation to Factory
                component = ComponentFactory.create(component_type, component_params)
                self.pipeline.append(component)
                
                logger.info(f"Pipeline built: Added component '{component_type}'.")
            except ValueError as e:
                logger.error(f"Failed to create component from config step: {e}")
                raise RuntimeError(f"Pipeline building failed: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during component creation: {e}")
                raise

    def fit(self, X: Any, y: Optional[Any] = None) -> 'ComponentOrchestrator':
        """
        Sequentially fits all components in the pipeline that require fitting (Stateful Adapters).
        """
        logger.info("Starting pipeline fitting...")
        for i, component in enumerate(self.pipeline):
            step_type = self.config[i].get("type")
            try:
                # Call fit, which might be a no-op for stateless components (like Resizer)
                component.fit(X, y)
                logger.debug(f"Successfully fitted component '{step_type}'.")
            except Exception as e:
                logger.error(f"Failed to fit component '{step_type}': {e}")
                raise
        return self

    def transform(self, X: Any) -> Any:
        """
        Sequentially applies transformations of all components in the pipeline.

        Args:
            X (Any): The input data.

        Returns:
            Any: The transformed data.
        """
        processed_data = X
        logger.info("Starting pipeline transformation...")
        for i, component in enumerate(self.pipeline):
            step_type = self.config[i].get("type")
            try:
                # Transformation is applied sequentially
                processed_data = component.transform(processed_data)
                logger.debug(f"Successfully transformed data with component '{step_type}'.")
            except Exception as e:
                logger.error(f"Failed to apply transformation step '{step_type}': {e}")
                raise
        return processed_data

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Combines fit and transform steps efficiently.
        """
        self.fit(X, y)
        return self.transform(X)

    def save(self, directory_path: str) -> None:
        """
        Saves the state of the entire pipeline, including learned parameters from stateful components.
        
        Args:
            directory_path (str): The directory where component states will be saved.
        """
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Saving pipeline state to {directory_path}...")
        
        for i, component in enumerate(self.pipeline):
            step_type = self.config[i].get("type")
            # Create a unique path for each component artifact
            component_path = os.path.join(directory_path, f"{i}_{step_type}_state.pkl")
            
            # Delegation to the component's save method
            component.save(component_path)
            logger.debug(f"Component '{step_type}' state saved.")

        logger.info("Full pipeline state saved successfully.")

    def load(self, directory_path: str) -> None:
        """
        Loads the state of the entire pipeline, re-initializing fitted components.
        
        Args:
            directory_path (str): The directory where component states are located.
        """
        logger.info(f"Loading pipeline state from {directory_path}...")
        
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Pipeline state directory not found: {directory_path}")

        for i, component in enumerate(self.pipeline):
            step_type = self.config[i].get("type")
            component_path = os.path.join(directory_path, f"{i}_{step_type}_state.pkl")
            
            # Only load state if the file exists (Stateless components like Resizer might not save a state)
            if os.path.exists(component_path):
                # Delegation to the component's load method
                component.load(component_path)
                logger.debug(f"Component '{step_type}' state loaded.")
            else:
                logger.warning(f"State file not found for component '{step_type}'. Assuming stateless.")

        logger.info("Full pipeline state loaded successfully.")