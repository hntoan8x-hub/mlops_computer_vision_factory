# cv_factory/shared_libs/ml_core/pipeline_components_cv/_feature_extractor/cv_orb_extractor.py (FIXED)

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union, List, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.feature_extractors.atomic.orb_extractor import ORBExtractor 

logger = logging.getLogger(__name__)

class CVORBExtractor(BaseComponent):
    """
    Adapter component for ORB (Oriented FAST and Rotated BRIEF) feature extraction.
    
    Manages ORB configuration and delegates execution to the atomic ORBExtractor. 
    This is a Stateless (non-learning) component.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, nfeatures: int = 500, scaleFactor: float = 1.2, nlevels: int = 8):
        """
        Initializes the Adapter and the Atomic ORB Extractor.

        Args:
            nfeatures (int): Maximum number of features to retain.
            scaleFactor (float): Pyramid decimation ratio.
            nlevels (int): The number of layers in the feature pyramid.
        """
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        
        self.atomic_extractor = ORBExtractor(
            nfeatures=self.nfeatures, 
            scaleFactor=self.scaleFactor, 
            nlevels=self.nlevels
        )
        
        logger.info(f"Initialized CVORBExtractor Adapter with nfeatures={self.nfeatures}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVORBExtractor':
        """
        CVORBExtractor is stateless, no fitting required.
        
        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVORBExtractor: The component instance.
        """
        logger.info("CVORBExtractor is stateless, no fitting required.")
        return self

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Extracts ORB features by delegating execution to the atomic extractor.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The extracted ORB features.
        """
        return self.atomic_extractor.extract(X) 

    def save(self, path: str) -> None:
        """
        Saves the component's configurable state (parameters) for persistence.

        Args:
            path (str): The path to save the state dictionary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'nfeatures': self.nfeatures, 
            'scaleFactor': self.scaleFactor, 
            'nlevels': self.nlevels
        }
        save_artifact(state, path)

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic extractor.

        Args:
            path (str): The path to the saved state.
        """
        state = load_artifact(path)
            
        self.nfeatures = state['nfeatures']
        self.scaleFactor = state['scaleFactor']
        self.nlevels = state['nlevels']
        
        self.atomic_extractor = ORBExtractor(
            nfeatures=self.nfeatures, 
            scaleFactor=self.scaleFactor, 
            nlevels=self.nlevels
        )