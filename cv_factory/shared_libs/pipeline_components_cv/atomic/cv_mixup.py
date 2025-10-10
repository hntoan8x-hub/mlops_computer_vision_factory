# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_mixup.py

import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union, Tuple

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# CRITICAL: Import the Atomic Logic Class (the Adaptee)
from shared_libs.data_processing.augmenters.atomic.mixup import Mixup 
# We assume Mixup is the class containing the mixing logic.

logger = logging.getLogger(__name__)

class CVMixup(BaseComponent):
    """
    Adapter component for applying Mixup augmentation. 
    
    This technique mixes two samples and their labels, requiring both X and Y inputs 
    in the transform method, which is common for batch augmentation.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initializes the Adapter and the Atomic Mixup logic.
        """
        self.alpha = alpha
        
        # Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_augmenter = Mixup(alpha=self.alpha)
        
        logger.info("Initialized CVMixup Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVMixup':
        logger.info("CVMixup is typically stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Applies Mixup by delegating execution to the atomic augmenter.

        Args:
            X (np.ndarray): The batch of input images.
            y (Optional[Any]): The batch of target labels.

        Returns:
            Tuple[np.ndarray, Optional[Any]]: The mixed images and labels.
        """
        if y is None:
            raise ValueError("Mixup requires labels (y) for label mixing.")
        
        # <<< ADAPTER LOGIC: Delegation of transformation >>>
        # Assuming the atomic layer has a transform method that handles both X and Y
        return self.atomic_augmenter.transform(X, y)
        # End of Adapter Logic
        
    def save(self, path: str) -> None:
        state = {'alpha': self.alpha}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVMixup state saved to {path}.")

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.alpha = state['alpha']
        self.atomic_augmenter = Mixup(alpha=self.alpha)
        logger.info(f"CVMixup state loaded from {path}.")