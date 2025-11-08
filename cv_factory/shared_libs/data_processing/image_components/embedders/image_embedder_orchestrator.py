# shared_libs/data_processing/image_components/embedders/image_embedder_orchestrator.py (HARDENED & DECOUPLED)

import logging
import numpy as np
import os
from typing import Dict, Any, List, Optional, Union

# --- Import Contracts and Configuration ---
from shared_libs.data_processing._base.base_embedder import BaseEmbedder, EmbeddingData
from shared_libs.data_processing.configs.preprocessing_config_schema import FeatureExtractionConfig

# --- Import Factory (CRITICAL FIX: Use the Embedder Factory for DI) ---
from shared_libs.data_processing.image_components.embedders.embedder_factory import EmbedderFactory 

ImageData = Union[np.ndarray, List[np.ndarray]]
logger = logging.getLogger(__name__)

class ImageEmbedderOrchestrator: # Renamed for clarity
    """
    Orchestrates and executes a sequence of deep learning embedding steps.

    This class acts as the decoupled, self-contained execution engine for the 
    embedding pipeline, managing component instantiation and sequential execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator by validating configuration and constructing the pipeline.

        Args:
            config (Dict[str, Any]): A dictionary containing the embedding configuration (FeatureExtractionConfig structure).

        Raises:
            ValueError: If the configuration fails Pydantic validation.
        """
        try:
            # 1. Configuration Validation (Quality Gate)
            # NOTE: We assume embedding steps are managed under FeatureExtractionConfig schema structure.
            self.config_schema: FeatureExtractionConfig = FeatureExtractionConfig(**config)
        except Exception as e:
            logger.error(f"FeatureExtractionConfig Validation Failed. Error: {e}")
            raise ValueError(f"Invalid FeatureExtraction Configuration provided: {e}")
            
        # 2. Pipeline Construction (Self-contained Execution Engine)
        self._pipeline: List[BaseEmbedder] = []
        for step in self.config_schema.components: # NOTE: Schema uses 'components'
            # For a dedicated EmbedderOrchestrator, we should filter or assume only embedder types are present.
            if step.enabled and ('embedder' in step.type.lower()):
                component = EmbedderFactory.create(
                    embedder_type=step.type,
                    config=step.params
                )
                self._pipeline.append(component)

        logger.info(f"ImageEmbedderOrchestrator initialized with {len(self._pipeline)} active steps.")

    def embed(self, data: ImageData, **kwargs: Dict[str, Any]) -> EmbeddingData:
        """
        Sequentially applies embedding extraction to the input image data.

        Args:
            data (ImageData): Preprocessed image data (expected to be a List[np.ndarray] or single np.ndarray).
            **kwargs: Additional keyword arguments.

        Returns:
            EmbeddingData: The final extracted embedding vector(s).
        """
        current_data = data
        for component in self._pipeline:
            # Execution logic: Call .embed()
            current_data = component.embed(current_data, **kwargs)
            logger.debug(f"Executed embedder: {component.__class__.__name__}")
            
        return current_data

    def fit(self, X: ImageData, y: Optional[Any] = None) -> 'ImageEmbedderOrchestrator':
        """
        Fits stateful components in the pipeline (e.g., fine-tuning layers or BatchNorm stats).

        Args:
            X (ImageData): Input image data used for fitting.
            y (Optional[Any]): Target data (optional).

        Returns:
            ImageEmbedderOrchestrator: The fitted orchestrator instance.
        """
        for step in self._pipeline:
            if hasattr(step, 'fit'):
                step.fit(X, y) 
        logger.info("Embedding pipeline fitting completed.")
        return self

    def save(self, directory_path: str) -> None:
        """
        Saves the state of the entire embedding pipeline by delegating saving to each atomic embedder.

        Args:
            directory_path (str): The file path where the pipeline state will be saved.
        """
        os.makedirs(directory_path, exist_ok=True)
        for i, step in enumerate(self._pipeline):
            try:
                # File name includes index and class name for safety
                file_name = f"{step.__class__.__name__}_{i}.pt" # Use .pt for PyTorch models
                if hasattr(step, 'save'):
                    step.save(os.path.join(directory_path, file_name))
            except Exception as e:
                logger.error(f"Error saving state for {step.__class__.__name__}: {e}")
                raise
        logger.info(f"Embedding pipeline state saved to {directory_path}.")

    def load(self, directory_path: str) -> None:
        """
        Loads the state of the entire embedding pipeline, re-initializing components based on the config.

        Args:
            directory_path (str): The file path from which the pipeline state will be loaded.
        """
        # Reconstruct the pipeline based on current config first (CRITICAL)
        self._pipeline = []
        for i, step_config in enumerate(self.config_schema.components):
            if step_config.enabled and ('embedder' in step_config.type.lower()):
                component = EmbedderFactory.create(
                    embedder_type=step_config.type, 
                    config=step_config.params
                )
                
                file_name = f"{component.__class__.__name__}_{i}.pt"
                file_path = os.path.join(directory_path, file_name)
                
                # Only load if the file exists and the component supports loading state
                if os.path.exists(file_path) and hasattr(component, 'load'):
                    component.load(file_path)
                
                self._pipeline.append(component)
                
        logger.info(f"Embedding pipeline state loaded from {directory_path}.")