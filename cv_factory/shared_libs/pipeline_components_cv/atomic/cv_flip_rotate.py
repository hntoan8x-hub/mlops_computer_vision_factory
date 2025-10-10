# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_flip_rotate.py

import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# CRITICAL: Import the Atomic Logic Class (the Adaptee)
from shared_libs.data_processing.augmenters.atomic.flip_rotate import FlipRotate 

logger = logging.getLogger(__name__)

class CVFlipRotate(BaseComponent):
    """
    Adapter component for flipping and rotating images.
    
    Adheres to BaseComponent and delegates execution to the atomic FlipRotate class.
    """
    
    def __init__(self, flip_prob: float = 0.5, rotate_limit: int = 15):
        """
        Initializes the Adapter and the Atomic Augmenter.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility)
        self.flip_prob = flip_prob
        self.rotate_limit = rotate_limit
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_augmenter = FlipRotate(
            flip_prob=self.flip_prob, 
            rotate_limit=self.rotate_limit
        )
        
        logger.info("Initialized CVFlipRotate Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVFlipRotate':
        logger.info("CVFlipRotate is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies augmentation by delegating execution to the atomic augmenter.
        """
        # <<< ADAPTER LOGIC: Delegation of transformation >>>
        return self.atomic_augmenter.transform(X) 
        # End of Adapter Logic

    def save(self, path: str) -> None:
        """
        Saves the component's state (parameters).
        """
        state = {'flip_prob': self.flip_prob, 'rotate_limit': self.rotate_limit}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVFlipRotate state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic augmenter.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.flip_prob = state['flip_prob']
        self.rotate_limit = state['rotate_limit']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_augmenter = FlipRotate(
            flip_prob=self.flip_prob, 
            rotate_limit=self.rotate_limit
        )
        logger.info(f"CVFlipRotate state loaded from {path}.")