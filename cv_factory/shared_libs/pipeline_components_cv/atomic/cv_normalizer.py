# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_normalizer.py

import logging
import numpy as np
from typing import Dict, Any, Union, Optional, List
import pickle

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# <<< BẮT BUỘC IMPORT ADAPTEE (Atomic Logic Class) >>>
from shared_libs.data_processing.cleaners.atomic.normalize_cleaner import NormalizeCleaner 
# Giả định NormalizeCleaner tồn tại và có phương thức transform(image)

logger = logging.getLogger(__name__)

class CVNormalizer(BaseComponent):
    """
    Adapter component for image pixel normalization. 
    
    It adheres to the BaseComponent contract, manages the mean/std state, and 
    delegates the actual transformation to the atomic NormalizeCleaner class.
    """
    
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        """
        Initializes the CVNormalizer Adapter.

        Args:
            mean (Union[float, List[float]]): The mean value(s) for normalization.
            std (Union[float, List[float]]): The standard deviation value(s) for normalization.
        """
        # 1. Manage State/Parameters (Responsibility of the Adapter)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        # Note: We instantiate the logic here, using the parameters managed by this Adapter
        self.atomic_cleaner = NormalizeCleaner(mean=mean, std=std) 
        
        logger.info(f"Initialized CVNormalizer Adapter. Atomic Cleaner created.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVNormalizer':
        """
        CVNormalizer is stateless (its parameters are pre-defined), so no fitting is required.
        """
        logger.info("CVNormalizer is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies normalization by delegating the call to the atomic NormalizeCleaner.
        """
        # <<< ADAPTER LOGIC HERE: Delegating execution to the Atomic Layer >>>
        return self.atomic_cleaner.transform(X)
        # End of Adapter Logic

    def save(self, path: str) -> None:
        """
        Saves the component's state (mean and std).
        """
        state = {'mean': self.mean, 'std': self.std}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVNormalizer state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic cleaner.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.mean = state['mean']
        self.std = state['std']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_cleaner = NormalizeCleaner(mean=self.mean, std=self.std) 
        logger.info(f"CVNormalizer state loaded from {path}.")