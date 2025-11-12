# shared_libs/ml_core/pipeline_components_cv/base_domain_adapter.py (UPDATED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import abc
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent

logger = logging.getLogger(__name__)

class BaseDomainAdapter(BaseComponent):
    """
    Abstract Base Adapter for multi-domain processing flows (Depth, Mask, Pointcloud, Text).

    This class delegates MLOps methods (fit/transform/save/load) to a specialized 
    Orchestrator/Processor (e.g., DepthProcessingOrchestrator) from the data_processing/ layer.
    """

    # Inherits REQUIRES_TARGET_DATA = False from BaseComponent.

    def __init__(self, processor: Any, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter.

        Args:
            processor (Any): Instance of the actual processor/orchestrator.
            name (str): Name of the Adapter (e.g., 'CVDepthCleaner').
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        self.processor = processor
        self.name = name
        self.config = config or {}
        logger.debug(f"Initialized BaseDomainAdapter for {name}")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'BaseComponent':
        """
        Delegates the fitting process to the specialized processor/orchestrator.

        Args:
            X (Any): Input data.
            y (Optional[Any]): Target data.

        Returns:
            BaseComponent: The fitted component instance.
        """
        if hasattr(self.processor, 'fit'):
            logger.info(f"Delegating fit() to underlying processor: {self.processor.__class__.__name__}")
            self.processor.fit(X, y)
        return self

    # Cập nhật: Sử dụng logic REQUIRES_TARGET_DATA để truyền y cho processor
    def transform(self, X: Any, y: Optional[Any] = None) -> Union[Any, Tuple[Any, Any]]:
        """
        Delegates the transformation/run process to the specialized processor/orchestrator.
        
        Args:
            X (Any): Input data to be transformed.
            y (Optional[Any]): Target data to be transformed.

        Returns:
            Union[Any, Tuple[Any, Any]]: The transformed data.
        """
        if hasattr(self.processor, 'run'):
            # Prefer 'run' (typical for Orchestrators)
            if self.REQUIRES_TARGET_DATA:
                return self.processor.run(X, y)
            return self.processor.run(X) 
        elif hasattr(self.processor, 'transform'):
            # Fallback to 'transform' (typical for simple Processors)
            if self.REQUIRES_TARGET_DATA:
                return self.processor.transform(X, y)
            return self.processor.transform(X)
        else:
            logger.error(f"Processor {self.processor.__class__.__name__} lacks required 'run' or 'transform' method.")
            raise NotImplementedError(f"Processor {self.processor.__class__.__name__} cannot execute transformation.")

    # fit_transform uses the default implementation from BaseComponent: fit(X, y).transform(X, y).

    def save(self, path: str) -> None:
        """
        Delegates saving the state.
        
        Args:
            path (str): The path to save the state.
        """
        if hasattr(self.processor, 'save'):
            self.processor.save(path)
        else:
            logger.debug(f"Processor {self.processor.__class__.__name__} is stateless; skipping save to {path}.")
            pass

    def load(self, path: str) -> None:
        """
        Delegates loading the state.
        
        Args:
            path (str): The path to the saved state.
        """
        if hasattr(self.processor, 'load'):
            self.processor.load(path)
        else:
            logger.debug(f"Processor {self.processor.__class__.__name__} is stateless; skipping load from {path}.")
            pass