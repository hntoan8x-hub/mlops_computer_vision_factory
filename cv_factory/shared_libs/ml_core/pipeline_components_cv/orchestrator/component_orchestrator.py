# cv_factory/shared_libs/ml_core/pipeline_components_cv/orchestrator/component_orchestrator.py (UPDATED)

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import copy 

# Import core components and factory
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory
from shared_libs.ml_core.pipeline_components_cv.configs.component_config_schema import PipelineStepConfig 

logger = logging.getLogger(__name__)

PipelineConfig = List[Dict[str, Any]]

class ComponentOrchestrator:
    """
    Execution Engine that orchestrates a sequence of Computer Vision pipeline components (Adapters).

    Manages pipeline building, sequential execution, and the MLOps lifecycle (fit, save, load).
    """

    def __init__(self, config: PipelineConfig, validator: Optional[Any] = None):
        """
        Initializes the orchestrator by building the pipeline.

        Args:
            config (PipelineConfig): Ordered list of component configuration dictionaries.
            validator (Optional[Any]): An optional validator object (Placeholder).
        
        Raises:
            RuntimeError: If pipeline building fails.
        """
        self.config: PipelineConfig = config
        self.validator = validator
        self.pipeline: List[BaseComponent] = []
        self.enabled_config_list: List[Dict[str, Any]] = [] 
        
        logger.info("Initializing ComponentOrchestrator...")
        self._build_pipeline()
        logger.info(f"Pipeline built successfully with {len(self.pipeline)} active steps.")

    def _build_pipeline(self) -> None:
        """
        Builds the sequence of pipeline components using the ComponentFactory.
        Also constructs self.enabled_config_list.
        """
        for i, step_config in enumerate(self.config):
            try:
                validated_config = PipelineStepConfig.model_validate(step_config)
            except Exception as e:
                logger.error(f"Configuration validation failed for step {i}: {e}")
                raise RuntimeError(f"Pipeline building failed due to invalid configuration at step {i}: {e}") from e
                
            if not validated_config.enabled:
                logger.info(f"Skipping disabled component at step {i} ('{validated_config.type}').")
                continue

            component_type = validated_config.type
            params = validated_config.params or {}

            try:
                # Use ComponentFactory.create (not create_component)
                component = ComponentFactory.create(
                    component_type=component_type, 
                    config=params
                )
                self.pipeline.append(component)
                self.enabled_config_list.append(step_config) 
                logger.debug(f"Successfully created and added component: {component_type}")
            except Exception as e:
                logger.error(f"Failed to instantiate component '{component_type}' at step {i}: {e}")
                raise RuntimeError(f"Pipeline building failed at step {i} ({component_type}): {e}") from e

    def _get_step_info(self, step_index: int) -> Tuple[str, str]:
        """
        Retrieves the class name and config type for a pipeline step.
        
        Args:
            step_index (int): Index of the step in the active pipeline.
            
        Returns:
            Tuple[str, str]: (Component Class Name, Config Type string).
        """
        component = self.pipeline[step_index]
        step_type = self.enabled_config_list[step_index].get("type", "unknown_config_type")
        return component.__class__.__name__, step_type

    def run(self, input_data: Any, target_data: Optional[Any] = None) -> Union[Any, Tuple[Any, Any]]:
        """
        Executes the pipeline sequentially (transform phase) using the static contract.

        Args:
            input_data (Any): The data (X) to be transformed.
            target_data (Optional[Any]): Target data (Y) to be transformed.

        Returns:
            Union[Any, Tuple[Any, Any]]: The final transformed data (X), or a tuple (X, Y).
        """
        logger.info(f"Starting sequential transformation run with {len(self.pipeline)} active components.")
        X = input_data
        Y = target_data
        
        for i, component in enumerate(self.pipeline):
            component_name, step_type = self._get_step_info(i)
            logger.debug(f"Executing transform on step {i}: {component_name} ({step_type})")
            
            try:
                is_xy_component = component.REQUIRES_TARGET_DATA

                if is_xy_component and Y is None:
                    # Enforce contract for components requiring Y
                    raise ValueError(f"Component '{step_type}' at step {i} requires target data (Y) but received None.")
                
                # All components now call component.transform(X, Y)
                result = component.transform(X, Y) 
                    
                if is_xy_component and isinstance(result, tuple) and len(result) == 2:
                    X, Y = result
                else:
                    # Most components (Cleaners, Embedders) return only X'
                    X = result
                    
            except Exception as e:
                logger.error(f"Execution failed at step {i} ({step_type}): {e}", exc_info=True)
                raise RuntimeError(f"Pipeline run failed at step {i} ({step_type}): {e}") from e
                
        logger.info("Sequential transformation run completed.")
        
        return (X, Y) if target_data is not None else X


    def fit(self, X: Any, y: Optional[Any] = None) -> 'ComponentOrchestrator':
        """
        Fits the entire pipeline sequentially with training data.
        
        Args:
            X (Any): Training data for fitting.
            y (Optional[Any]): Training labels.

        Returns:
            ComponentOrchestrator: The fitted orchestrator instance.
        """
        logger.info(f"Starting sequential fitting of {len(self.pipeline)} active components.")
        
        current_X = X
        current_y = y
        
        for i, component in enumerate(self.pipeline):
            component_name, step_type = self._get_step_info(i)
            logger.debug(f"Executing fit_transform on step {i}: {component_name} ({step_type})")

            try:
                # fit_transform is now reliably implemented by BaseComponent (or overridden for optimization)
                result = component.fit_transform(current_X, current_y)
                
                # Apply result: If REQUIRES_TARGET_DATA is True and returns a tuple, update both X and Y
                if component.REQUIRES_TARGET_DATA and isinstance(result, tuple) and len(result) == 2:
                    current_X, current_y = result
                else:
                    # Most components (fit and return only X')
                    current_X = result
                    
            except Exception as e:
                logger.error(f"Fitting failed at step {i} ({step_type}): {e}", exc_info=True)
                raise RuntimeError(f"Pipeline fitting failed at step {i} ({step_type}): {e}") from e
                
        logger.info("Sequential pipeline fitting completed.")
        return self

    def save(self, directory_path: str) -> None:
        """
        Delegates saving the state of the entire preprocessing pipeline.
        
        Args:
            directory_path (str): Directory where component states will be saved.
        
        Raises:
            RuntimeError: If saving fails (including NotImplementedError).
        """
        logger.info(f"Saving pipeline state to {directory_path}...")
        
        os.makedirs(directory_path, exist_ok=True)
        
        for i, component in enumerate(self.pipeline):
            component_name, step_type = self._get_step_info(i)
            component_path = os.path.join(directory_path, f"{i}_{step_type}_state.pkl")
            
            try:
                # Adapter phải triển khai save. Nếu không, NotImplementedError sẽ bị bắt bên dưới.
                component.save(component_path) 
                logger.debug(f"Component '{step_type}' state saved to {component_path}.")
            except Exception as e:
                # Tất cả các lỗi (I/O, Serialization, NotImplementedError) đều là lỗi nghiêm trọng.
                logger.error(f"Failed to save state for component '{step_type}' at step {i}: {e}", exc_info=True)
                raise RuntimeError(f"Pipeline save failed at step {i} ({step_type}): {e}")

        logger.info("Full pipeline state saved successfully.")

    def load(self, directory_path: str) -> None:
        """
        Delegates loading the state of the entire preprocessing pipeline.
        
        Args:
            directory_path (str): Directory containing component states.
        
        Raises:
            FileNotFoundError: If the state directory does not exist.
            RuntimeError: If loading fails (including NotImplementedError).
        """
        logger.info(f"Loading full preprocessing pipeline state from {directory_path}...")
        
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Pipeline state directory not found: {directory_path}")

        for i, component in enumerate(self.pipeline):
            component_name, step_type = self._get_step_info(i)
            component_path = os.path.join(directory_path, f"{i}_{step_type}_state.pkl")
            
            if os.path.exists(component_path):
                try:
                    # Adapter phải triển khai load.
                    component.load(component_path)
                    logger.debug(f"Component '{step_type}' state loaded from {component_path}.")
                except Exception as e:
                    # Tất cả các lỗi (I/O, Deserialization, NotImplementedError) đều là lỗi nghiêm trọng.
                    logger.error(f"Failed to load state for component '{step_type}' at step {i}: {e}", exc_info=True)
                    raise RuntimeError(f"Pipeline load failed at step {i} ({step_type}): {e}")
            else:
                # OK nếu file state không tồn tại, Adapter sẽ dùng __init__
                logger.warning(f"State file not found for component '{step_type}'. Assuming default initialization.")

        logger.info("Full pipeline state loaded successfully.")