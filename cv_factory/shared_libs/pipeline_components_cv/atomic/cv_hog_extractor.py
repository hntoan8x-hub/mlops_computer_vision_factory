# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_hog_extractor.py

import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union, Tuple, List

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# CRITICAL: Import the Atomic Logic Class (the Adaptee)
from shared_libs.data_processing.feature_extractors.atomic.hog_extractor import HOGExtractor 

logger = logging.getLogger(__name__)

class CVHOGExtractor(BaseComponent):
    """
    Adapter component for Histogram of Oriented Gradients (HOG) feature extraction.
    
    Adheres to BaseComponent, manages HOG parameters, and delegates execution.
    """
    
    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8), 
                 cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Initializes the Adapter and the Atomic HOG Extractor.
        """
        # 1. Manage State/Parameters (Adapter's Responsibility)
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_extractor = HOGExtractor(
            orientations=self.orientations, 
            pixels_per_cell=self.pixels_per_cell, 
            cells_per_block=self.cells_per_block
        )
        
        logger.info("Initialized CVHOGExtractor Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVHOGExtractor':
        logger.info("CVHOGExtractor is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts HOG features by delegating execution to the atomic extractor.
        """
        # <<< ADAPTER LOGIC: Delegation of extraction >>>
        return self.atomic_extractor.extract(X) 
        # End of Adapter Logic

    def save(self, path: str) -> None:
        """
        Saves the component's state (parameters).
        """
        state = {
            'orientations': self.orientations, 
            'pixels_per_cell': self.pixels_per_cell, 
            'cells_per_block': self.cells_per_block
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVHOGExtractor state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic extractor.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.orientations = state['orientations']
        self.pixels_per_cell = state['pixels_per_cell']
        self.cells_per_block = state['cells_per_block']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_extractor = HOGExtractor(
            orientations=self.orientations, 
            pixels_per_cell=self.pixels_per_cell, 
            cells_per_block=self.cells_per_block
        )
        logger.info(f"CVHOGExtractor state loaded from {path}.")