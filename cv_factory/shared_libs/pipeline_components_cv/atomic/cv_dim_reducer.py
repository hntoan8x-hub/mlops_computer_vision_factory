# shared_libs/ml_core/pipeline_components_cv/atomic/cv_dim_reducer.py

import logging
import numpy as np
import pickle
from typing import Dict, Any, Union, List, Optional

# --- Import Abstraction & Adaptee ---
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# The Atomic Logic Class (The Adaptee)
from shared_libs.data_processing.image_components.feature_extractors.atomic.dim_reducer_atomic import DimReducerAtomic 

logger = logging.getLogger(__name__)

class CVDimReducer(BaseComponent):
    """
    Adapter component for dimensionality reduction of feature vectors (e.g., PCA, UMAP).
    
    This class is STATEFUL: it implements fit(), save(), and load() to persist 
    the parameters learned during the fitting phase (e.g., the PCA components).
    It wraps the stateless (but stateful in operation) DimReducerAtomic class.
    """
    
    def __init__(self, method: str, n_components: int = 2):
        """
        Initializes the CVDimReducer Adapter.

        Args:
            method (str): The dimensionality reduction method ('pca', 'umap', etc.).
            n_components (int): The target number of dimensions.
        
        Raises:
            ValueError: If the reduction method is unsupported by DimReducerAtomic.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility for MLOps)
        self.method = method
        self.n_components = n_components
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        # This Adaptee class handles the actual math and internal model creation.
        self.atomic_reducer = DimReducerAtomic(method=self.method, n_components=self.n_components)
        
        logger.info(f"Initialized CVDimReducer Adapter for method '{self.method}'.")

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> 'CVDimReducer':
        """
        Fits the dimensionality reduction model to the data by delegating the call 
        to the atomic logic layer (DimReducerAtomic).
        
        Args:
            X (np.ndarray): Input feature vectors (n_samples, n_features).
            y (Optional[Any]): Target data (optional).
            
        Returns:
            CVDimReducer: The fitted component instance.
        """
        # ADAPTER LOGIC: Delegation of fitting
        self.atomic_reducer.fit(X, y)
        logger.info(f"CVDimReducer has been fitted using {self.method}.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fitted transformation to the input data.

        Args:
            X (np.ndarray): Input data to be transformed.

        Returns:
            np.ndarray: The dimensionality-reduced feature vectors.
        """
        # ADAPTER LOGIC: Delegation of transformation
        return self.atomic_reducer.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Combines fit and transform steps (Delegated).

        Args:
            X (np.ndarray): Input data.
            y (Optional[Any]): Target data.

        Returns:
            np.ndarray: The transformed data.
        """
        # Rely on the Adaptee to implement the fit_transform efficiency
        return self.atomic_reducer.fit_transform(X, y) 

    def save(self, path: str) -> None:
        """
        Saves the fitted state of the atomic reducer model (e.g., the fitted PCA object).

        Args:
            path (str): The path to save the state file.
        """
        # Delegation: Save the internal, fitted state of the Adaptee.
        self.atomic_reducer.save(path)
        logger.info(f"CVDimReducer state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the fitted state and re-initializes the atomic reducer.

        Args:
            path (str): The path to the saved state file.
        """
        # Delegation: Load the internal, fitted state into the Adaptee.
        self.atomic_reducer.load(path)
        
        # NOTE: The instance's parameters (method, n_components) should ideally be verified 
        # against the loaded state's parameters for full robustness.
        logger.info(f"CVDimReducer state loaded from {path}.")