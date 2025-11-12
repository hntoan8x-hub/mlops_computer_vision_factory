# shared_libs/ml_core/pipeline_components_cv/_cleaner/cv_normalizer.py (UPDATED)

import logging
import numpy as np
from typing import Dict, Any, Union, Optional, List, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.cleaners.atomic.normalize_cleaner import NormalizeCleaner # Atomic Logic

logger = logging.getLogger(__name__)

class CVNormalizer(BaseComponent):
    """
    Adapter component for image pixel normalization. 
    
    Manages the mean/std configuration, implements persistence, and delegates transformation 
    to the atomic NormalizeCleaner. This component is Stateless (non-learning).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        """
        Initializes the CVNormalizer Adapter.

        Args:
            mean (Union[float, List[float]]): The mean value(s) for normalization.
            std (Union[float, List[float]]): The standard deviation value(s) for normalization.
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        # Instantiate the atomic logic with the configuration
        # Note: We pass the original List/float to the atomic cleaner as per its expected __init__
        self.atomic_cleaner = NormalizeCleaner(mean=mean, std=std) 
        
        logger.info(f"Initialized CVNormalizer Adapter. Atomic Cleaner created.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVNormalizer':
        """
        CVNormalizer is stateless (its parameters are pre-defined), so no fitting is required.

        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVNormalizer: The component instance.
        """
        logger.info("CVNormalizer is stateless, no fitting required.")
        return self

    # Sửa lỗi hợp đồng: Bắt buộc nhận y: Optional[Any] = None
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Applies normalization by delegating the call to the atomic NormalizeCleaner.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).

        Returns:
            np.ndarray: The normalized image array(s).
        """
        return self.atomic_cleaner.transform(X)

    def save(self, path: str) -> None:
        """
        Saves the component's configurable state (mean and std) for persistence.

        Args:
            path (str): The path to save the state dictionary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert back to list/float for simple serialization if necessary, or save arrays directly
        state = {'mean': self.mean, 'std': self.std}
        save_artifact(state, path)
        
    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic cleaner.

        Args:
            path (str): The path to the saved state.
        """
        state = load_artifact(path)
        
        self.mean = state['mean']
        self.std = state['std']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_cleaner = NormalizeCleaner(
            mean=self.mean, 
            std=self.std
        )