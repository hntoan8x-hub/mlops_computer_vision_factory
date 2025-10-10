# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_resizer.py

import logging
import numpy as np
import cv2
from typing import Dict, Any, Union, Optional
import pickle

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# <<< CRITICAL: Import the Atomic Logic Class (the Adaptee) >>>
from shared_libs.data_processing.cleaners.atomic.resize_cleaner import ResizeCleaner 
# We assume ResizeCleaner contains the actual cv2.resize implementation.

logger = logging.getLogger(__name__)

class CVResizer(BaseComponent):
    """
    Adapter component for resizing images. 
    
    It adheres to the BaseComponent contract, manages the resizing parameters, 
    and delegates the actual image transformation to the atomic ResizeCleaner class.
    """
    
    def __init__(self, width: int, height: int, interpolation: int = cv2.INTER_AREA):
        """
        Initializes the CVResizer Adapter.

        Args:
            width (int): The target width.
            height (int): The target height.
            interpolation (int): The interpolation method (e.g., cv2.INTER_AREA).
        """
        # 1. Manage State/Parameters (Responsibility of the Adapter)
        self.width = width
        self.height = height
        self.interpolation = interpolation
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_cleaner = ResizeCleaner(
            width=self.width, 
            height=self.height, 
            interpolation=self.interpolation
        )
        
        logger.info(f"Initialized CVResizer Adapter to {self.width}x{self.height}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVResizer':
        """
        CVResizer is a stateless component, no fitting is required.
        """
        logger.info("CVResizer is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies resizing by delegating the execution to the atomic ResizeCleaner.
        
        Args:
            X (np.ndarray): The input image array(s).

        Returns:
            np.ndarray: The resized image array(s).
        """
        # <<< ADAPTER LOGIC: Delegation to the Atomic Layer >>>
        return self.atomic_cleaner.transform(X)
        # End of Adapter Logic

    def save(self, path: str) -> None:
        """
        Saves the component's state (its parameters).
        """
        state = {
            'width': self.width, 
            'height': self.height, 
            'interpolation': self.interpolation
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVResizer state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic cleaner.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.width = state['width']
        self.height = state['height']
        self.interpolation = state['interpolation']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_cleaner = ResizeCleaner(
            width=self.width, 
            height=self.height, 
            interpolation=self.interpolation
        )
        logger.info(f"CVResizer state loaded from {path}.")