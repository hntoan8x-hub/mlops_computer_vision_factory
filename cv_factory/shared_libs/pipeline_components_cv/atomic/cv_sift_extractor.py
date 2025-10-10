# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_sift_extractor.py

import logging
import numpy as np
import cv2
import pickle
from typing import Dict, Any, Optional, Union, List

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# CRITICAL: Import the Atomic Logic Class (the Adaptee)
from shared_libs.data_processing.feature_extractors.atomic.sift_extractor import SIFTExtractor 

logger = logging.getLogger(__name__)

class CVSIFTExtractor(BaseComponent):
    """
    Adapter component for SIFT (Scale-Invariant Feature Transform) feature extraction.
    
    Adheres to BaseComponent, manages SIFT parameters (e.g., nfeatures), and delegates execution.
    """
    
    def __init__(self, nfeatures: int = 0, nOctaveLayers: int = 3, contrastThreshold: float = 0.04):
        """
        Initializes the Adapter and the Atomic SIFT Extractor.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility)
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_extractor = SIFTExtractor(
            nfeatures=self.nfeatures, 
            nOctaveLayers=self.nOctaveLayers, 
            contrastThreshold=self.contrastThreshold
        )
        
        logger.info(f"Initialized CVSIFTExtractor Adapter with nfeatures={self.nfeatures}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVSIFTExtractor':
        logger.info("CVSIFTExtractor is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts SIFT features by delegating execution to the atomic extractor.
        """
        # <<< ADAPTER LOGIC: Delegation of extraction >>>
        return self.atomic_extractor.extract(X) 

    def save(self, path: str) -> None:
        """Saves the component's state (parameters)."""
        state = {
            'nfeatures': self.nfeatures, 
            'nOctaveLayers': self.nOctaveLayers, 
            'contrastThreshold': self.contrastThreshold
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVSIFTExtractor state saved to {path}.")

    def load(self, path: str) -> None:
        """Loads the component's state and re-initializes the atomic extractor."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.nfeatures = state['nfeatures']
        self.nOctaveLayers = state['nOctaveLayers']
        self.contrastThreshold = state['contrastThreshold']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_extractor = SIFTExtractor(
            nfeatures=self.nfeatures, 
            nOctaveLayers=self.nOctaveLayers, 
            contrastThreshold=self.contrastThreshold
        )
        logger.info(f"CVSIFTExtractor state loaded from {path}.")