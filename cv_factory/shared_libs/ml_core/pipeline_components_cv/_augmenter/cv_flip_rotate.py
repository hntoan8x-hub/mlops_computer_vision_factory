# cv_factory/shared_libs/ml_core/pipeline_components_cv/_augmenter/cv_flip_rotate.py (FIXED)

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.augmenters.atomic.flip_rotate import FlipRotate 

logger = logging.getLogger(__name__)

class CVFlipRotate(BaseComponent):
    """
    Adapter component for flipping and rotating images.
    
    This is an X-only augmentation (REQUIRES_TARGET_DATA=False).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, flip_prob: float = 0.5, rotate_limit: int = 15):
        """
        Initializes the Adapter and the Atomic Augmenter.
        
        Args:
            flip_prob (float): Probability of applying flip.
            rotate_limit (int): The maximum rotation angle.
        """
        self.flip_prob = flip_prob
        self.rotate_limit = rotate_limit
        
        # Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_augmenter = FlipRotate(
            flip_prob=self.flip_prob, 
            rotate_limit=self.rotate_limit
        )
        
        logger.info("Initialized CVFlipRotate Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVFlipRotate':
        logger.info("CVFlipRotate is stateless, no fitting required.")
        return self

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Applies augmentation by delegating execution to the atomic augmenter.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The augmented image array(s).
        """
        # Atomic logic does not need y
        return self.atomic_augmenter.transform(X) 

    def save(self, path: str) -> None:
        """Saves the component's state (parameters) using the utility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {'flip_prob': self.flip_prob, 'rotate_limit': self.rotate_limit}
        save_artifact(state, path)

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic augmenter.
        """
        state = load_artifact(path)
            
        self.flip_prob = state['flip_prob']
        self.rotate_limit = state['rotate_limit']
        
        self.atomic_augmenter = FlipRotate(
            flip_prob=self.flip_prob, 
            rotate_limit=self.rotate_limit
        )