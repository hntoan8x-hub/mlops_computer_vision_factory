# cv_factory/shared_libs/ml_core/pipeline_components_cv/_feature_extractor/cv_dim_reducer.py (FIXED)

import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.feature_extractors.atomic.dim_reducer_atomic import DimReducerAtomic 

logger = logging.getLogger(__name__)

class CVDimReducer(BaseComponent):
    """
    Adapter component for dimensionality reduction of feature vectors (e.g., PCA, UMAP).
    
    This is a STATEFUL component: it implements fit(), save(), and load() to persist 
    the parameters learned during the fitting phase.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False (Unsupervised method)

    def __init__(self, method: str, n_components: int = 2):
        """
        Initializes the CVDimReducer Adapter.

        Args:
            method (str): The dimensionality reduction method ('pca', 'umap', etc.).
            n_components (int): The target number of dimensions.
        """
        self.method = method
        self.n_components = n_components
        
        self.atomic_reducer = DimReducerAtomic(method=self.method, n_components=self.n_components)
        
        logger.info(f"Initialized CVDimReducer Adapter for method '{self.method}'.")

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> 'CVDimReducer':
        """
        Fits the dimensionality reduction model to the data via delegation.
        
        Args:
            X (np.ndarray): Input feature vectors (n_samples, n_features).
            y (Optional[Any]): Target data (optional, used for supervised methods like Linear Discriminant Analysis).
            
        Returns:
            CVDimReducer: The fitted component instance.
        """
        self.atomic_reducer.fit(X, y)
        logger.info(f"CVDimReducer has been fitted using {self.method}.")
        return self

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Applies the fitted transformation to the input data.

        Args:
            X (np.ndarray): Input data to be transformed.
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The dimensionality-reduced feature vectors.
        """
        # Delegation: Atomic logic typically handles X only
        return self.atomic_reducer.transform(X)

    # FIX: Bỏ override fit_transform và sử dụng default implementation của BaseComponent 
    # (trừ khi có lý do tối ưu hóa hiệu suất rõ ràng)
    # Nếu muốn dùng lại logic cũ (dựa vào Adaptee):
    def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Combines fit and transform steps (Delegated).

        Args:
            X (np.ndarray): Input data.
            y (Optional[Any]): Target data.

        Returns:
            np.ndarray: The transformed data.
        """
        return self.atomic_reducer.fit_transform(X, y) 
    # Kết luận: Giữ lại override fit_transform nếu Adaptee đã tối ưu hóa, nhưng sửa signature.

    def save(self, path: str) -> None:
        """
        Saves the fitted state of the atomic reducer model via delegation.

        Args:
            path (str): The path to save the state.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Delegation: Save the internal, fitted state of the Adaptee.
        self.atomic_reducer.save(path)
        logger.info(f"CVDimReducer state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the fitted state and re-initializes the atomic reducer via delegation.

        Args:
            path (str): The path to the saved state.
        """
        # Delegation: Load the internal, fitted state into the Adaptee.
        self.atomic_reducer.load(path)
        logger.info(f"CVDimReducer state loaded from {path}.")