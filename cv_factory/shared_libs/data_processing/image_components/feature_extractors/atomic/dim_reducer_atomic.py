# cv_factory/shared_libs/data_processing/image_components/feature_extractors/atomic/dim_reducer_atomic.py

import logging
from typing import Dict, Any, Optional
import numpy as np
import pickle
from sklearn.decomposition import PCA
# NOTE: Assume UMAP is available for demonstration purposes
# import umap.UMAP as UMAP 

logger = logging.getLogger(__name__)

class DimReducerAtomic:
    """
    Atomic logic class for dimensionality reduction methods (e.g., PCA, UMAP).
    
    This class is the 'Adaptee' in the Adapter Pattern. It is responsible for 
    the actual mathematical execution and serialization of the fitted model.
    """
    
    def __init__(self, method: str, n_components: int = 2):
        """
        Initializes the atomic logic component.

        Args:
            method (str): The reduction method ('pca', 'umap').
            n_components (int): The target number of dimensions.
        """
        self.method = method
        self.n_components = n_components
        self.reducer = self._get_reducer(method, n_components)
        logger.info(f"DimReducerAtomic initialized with method '{self.method}'.")

    def _get_reducer(self, method: str, n_components: int) -> Any:
        """Helper to instantiate the appropriate reducer model."""
        method = method.lower()
        if method == "pca":
            return PCA(n_components=n_components)
        # elif method == "umap":
        #     return UMAP(n_components=n_components)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}.")

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> None:
        """
        Fits the internal model (PCA/UMAP) to the input data.
        """
        if len(X.shape) != 2:
            raise ValueError("Input data for dim reducer must be a 2D array of shape (n_samples, n_features).")
        self.reducer.fit(X, y)
        logger.debug("Atomic reducer has been fitted.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies a transformation to the input data.
        """
        return self.reducer.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Combines fit and transform steps.
        """
        # Rely on the underlying library's efficient implementation
        return self.reducer.fit_transform(X, y)

    def save(self, path: str) -> None:
        """
        Saves the state of the fitted reducer model using pickle.
        """
        # CRITICAL: This method handles the actual I/O serialization
        with open(path, 'wb') as f:
            pickle.dump(self.reducer, f)
        logger.info(f"Atomic reducer state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the fitted state of the reducer model.
        """
        # CRITICAL: This method handles the actual I/O deserialization
        with open(path, 'rb') as f:
            self.reducer = pickle.load(f)
        logger.info(f"Atomic reducer state loaded from {path}.")