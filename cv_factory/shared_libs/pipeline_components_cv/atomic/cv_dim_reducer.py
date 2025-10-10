# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_dim_reducer.py

import logging
import numpy as np
import pickle
from typing import Dict, Any, Union, List, Optional
# We will use the standard pickle for saving the fitted reducer object.

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# <<< CRITICAL: Import the Atomic Logic Class (the Adaptee) >>>
from shared_libs.data_processing.feature_extractors.atomic.dim_reducer_atomic import DimReducerAtomic 
# Assuming DimReducerAtomic is the class containing PCA/UMAP logic.

logger = logging.getLogger(__name__)

class CVDimReducer(BaseComponent):
    """
    Adapter component for dimensionality reduction of feature vectors (e.g., PCA, UMAP).
    
    This class is STATEFUL: it implements fit(), save(), and load() to persist 
    the parameters learned during the fitting phase (e.g., the PCA components).
    """
    
    def __init__(self, method: str, n_components: int = 2):
        """
        Initializes the CVDimReducer Adapter.

        Args:
            method (str): The dimensionality reduction method ('pca', 'umap', etc.).
            n_components (int): The target number of dimensions.
        """
        # 1. Manage State/Parameters
        self.method = method
        self.n_components = n_components
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        # This Adaptee class handles the actual math and internal model creation (PCA, UMAP)
        self.atomic_reducer = DimReducerAtomic(method=self.method, n_components=self.n_components)
        
        logger.info(f"Initialized CVDimReducer Adapter with method '{self.method}'.")

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> 'CVDimReducer':
        """
        Fits the dimensionality reduction model (e.g., PCA) to the data by 
        delegating the call to the atomic logic layer.
        
        Args:
            X (np.ndarray): Input feature vectors (n_samples, n_features).
        """
        # <<< ADAPTER LOGIC: Delegation of fitting >>>
        self.atomic_reducer.fit(X, y)
        logger.info(f"CVDimReducer has been fitted using {self.method}.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fitted transformation to the input data.
        """
        # <<< ADAPTER LOGIC: Delegation of transformation >>>
        return self.atomic_reducer.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Combines fit and transform steps (Delegated).
        """
        # We rely on the Adaptee to implement the fit_transform efficiency
        return self.atomic_reducer.fit_transform(X, y) 

    def save(self, path: str) -> None:
        """
        Saves the fitted state of the atomic reducer model (e.g., the fitted PCA object).
        """
        # CRITICAL: We save the internal, fitted state of the Adaptee.
        self.atomic_reducer.save(path)
        logger.info(f"CVDimReducer state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the fitted state and re-initializes the atomic reducer.
        """
        # CRITICAL: We load the internal, fitted state into the Adaptee.
        self.atomic_reducer.load(path)
        
        # The internal parameters (method, n_components) might also be loaded if needed, 
        # but primarily, the fitted model is loaded into the atomic_reducer instance.
        logger.info(f"CVDimReducer state loaded from {path}.")