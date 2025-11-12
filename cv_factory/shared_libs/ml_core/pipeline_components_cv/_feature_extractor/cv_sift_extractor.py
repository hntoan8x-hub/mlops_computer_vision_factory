# cv_factory/shared_libs/ml_core/pipeline_components_cv/_feature_extractor/cv_sift_extractor.py (FIXED)

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union, List, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.feature_extractors.atomic.sift_extractor import SIFTExtractor 

logger = logging.getLogger(__name__)

class CVSIFTExtractor(BaseComponent):
    """
    Adapter component for SIFT (Scale-Invariant Feature Transform) feature extraction.
    
    Manages SIFT configuration and delegates execution to the atomic SIFTExtractor. 
    This is a Stateless (non-learning) component.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, nfeatures: int = 0, nOctaveLayers: int = 3, contrastThreshold: float = 0.04):
        """
        Initializes the Adapter and the Atomic SIFT Extractor.

        Args:
            nfeatures (int): Maximum number of features to retain.
            nOctaveLayers (int): Number of layers in each octave.
            contrastThreshold (float): Contrast threshold for filtering keypoints.
        """
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        
        self.atomic_extractor = SIFTExtractor(
            nfeatures=self.nfeatures, 
            nOctaveLayers=self.nOctaveLayers, 
            contrastThreshold=self.contrastThreshold
        )
        
        logger.info(f"Initialized CVSIFTExtractor Adapter with nfeatures={self.nfeatures}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVSIFTExtractor':
        """
        CVSIFTExtractor is stateless, no fitting required.
        
        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVSIFTExtractor: The component instance.
        """
        logger.info("CVSIFTExtractor is stateless, no fitting required.")
        return self

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Extracts SIFT features by delegating execution to the atomic extractor.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The extracted SIFT features.
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
            'nOctaveLayers': self.nOctaveLayers, 
            'contrastThreshold': self.contrastThreshold
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
        self.nOctaveLayers = state['nOctaveLayers']
        self.contrastThreshold = state['contrastThreshold']
        
        self.atomic_extractor = SIFTExtractor(
            nfeatures=self.nfeatures, 
            nOctaveLayers=self.nOctaveLayers, 
            contrastThreshold=self.contrastThreshold
        )