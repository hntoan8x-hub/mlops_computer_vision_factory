# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_orb_extractor.py

import logging
import numpy as np
import cv2
import pickle
from typing import Dict, Any, Optional, Union, List

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# CRITICAL: Import the Atomic Logic Class (the Adaptee)
from shared_libs.data_processing.feature_extractors.atomic.orb_extractor import ORBExtractor 

logger = logging.getLogger(__name__)

class CVORBExtractor(BaseComponent):
    """
    Adapter component for ORB (Oriented FAST and Rotated BRIEF) feature extraction.
    
    Adheres to BaseComponent, manages ORB parameters (e.g., nfeatures), and delegates execution.
    """
    
    def __init__(self, nfeatures: int = 500, scaleFactor: float = 1.2, nlevels: int = 8):
        """
        Initializes the Adapter and the Atomic ORB Extractor.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility)
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_extractor = ORBExtractor(
            nfeatures=self.nfeatures, 
            scaleFactor=self.scaleFactor, 
            nlevels=self.nlevels
        )
        
        logger.info(f"Initialized CVORBExtractor Adapter with nfeatures={self.nfeatures}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVORBExtractor':
        logger.info("CVORBExtractor is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts ORB features by delegating execution to the atomic extractor.
        """
        # <<< ADAPTER LOGIC: Delegation of extraction >>>
        return self.atomic_extractor.extract(X) 

    def save(self, path: str) -> None:
        """Saves the component's state (parameters)."""
        state = {
            'nfeatures': self.nfeatures, 
            'scaleFactor': self.scaleFactor, 
            'nlevels': self.nlevels
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVORBExtractor state saved to {path}.")

    def load(self, path: str) -> None:
        """Loads the component's state and re-initializes the atomic extractor."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.nfeatures = state['nfeatures']
        self.scaleFactor = state['scaleFactor']
        self.nlevels = state['nlevels']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_extractor = ORBExtractor(
            nfeatures=self.nfeatures, 
            scaleFactor=self.scaleFactor, 
            nlevels=self.nlevels
        )
        logger.info(f"CVORBExtractor state loaded from {path}.")