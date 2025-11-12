# shared_libs/ml_core/pipeline_components_cv/_cleaner/cv_resizer.py (UPDATED)

import logging
import numpy as np
import cv2
from typing import Dict, Any, Union, Optional, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.cleaners.atomic.resize_cleaner import ResizeCleaner # Atomic Logic

logger = logging.getLogger(__name__)

class CVResizer(BaseComponent):
    """
    Adapter component for resizing images. 
    
    Manages resizing parameters, implements persistence, and delegates transformation
    to the atomic ResizeCleaner. This component is Stateless (non-learning).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, width: int, height: int, interpolation: int = cv2.INTER_AREA):
        """
        Initializes the CVResizer Adapter.

        Args:
            width (int): The target width.
            height (int): The target height.
            interpolation (int): The interpolation method (e.g., cv2.INTER_AREA).
        """
        self.width = width
        self.height = height
        self.interpolation = interpolation
        
        self.atomic_cleaner = ResizeCleaner(
            width=self.width, 
            height=self.height, 
            interpolation=self.interpolation
        )
        
        logger.info(f"Initialized CVResizer Adapter to {self.width}x{self.height}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVResizer':
        """
        CVResizer is a stateless component, no fitting is required.

        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVResizer: The component instance.
        """
        logger.info("CVResizer is stateless, no fitting required.")
        return self

    # Sửa lỗi hợp đồng: Bắt buộc nhận y: Optional[Any] = None
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Applies resizing by delegating the execution to the atomic ResizeCleaner.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).

        Returns:
            np.ndarray: The resized image array(s).
        """
        return self.atomic_cleaner.transform(X)

    def save(self, path: str) -> None:
        """
        Saves the component's configurable state (parameters) for persistence.

        Args:
            path (str): The path to save the state dictionary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'width': self.width, 
            'height': self.height, 
            'interpolation': self.interpolation
        }
        save_artifact(state, path)
        
    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic cleaner.

        Args:
            path (str): The path to the saved state.
        """
        state = load_artifact(path)
            
        self.width = state['width']
        self.height = state['height']
        self.interpolation = state['interpolation']
        
        self.atomic_cleaner = ResizeCleaner(
            width=self.width, 
            height=self.height, 
            interpolation=self.interpolation
        )