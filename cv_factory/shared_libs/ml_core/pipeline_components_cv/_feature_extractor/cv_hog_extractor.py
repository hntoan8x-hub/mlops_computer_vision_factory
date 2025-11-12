# cv_factory/shared_libs/ml_core/pipeline_components_cv/_feature_extractor/cv_hog_extractor.py (FIXED)

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.feature_extractors.atomic.hog_extractor import HOGExtractor 

logger = logging.getLogger(__name__)

class CVHOGExtractor(BaseComponent):
    """
    Adapter component for Histogram of Oriented Gradients (HOG) feature extraction.
    
    Manages HOG configuration and delegates execution to the atomic HOGExtractor. 
    This is a Stateless (non-learning) component.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8), 
                 cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Initializes the Adapter and the Atomic HOG Extractor.

        Args:
            orientations (int): Number of orientation bins.
            pixels_per_cell (Tuple[int, int]): Size of the cell in pixels.
            cells_per_block (Tuple[int, int]): Number of cells in each block.
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
        self.atomic_extractor = HOGExtractor(
            orientations=self.orientations, 
            pixels_per_cell=self.pixels_per_cell, 
            cells_per_block=self.cells_per_block
        )
        
        logger.info("Initialized CVHOGExtractor Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVHOGExtractor':
        """
        CVHOGExtractor is stateless, no fitting required.
        
        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVHOGExtractor: The component instance.
        """
        logger.info("CVHOGExtractor is stateless, no fitting required.")
        return self

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Extracts HOG features by delegating execution to the atomic extractor.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored).

        Returns:
            np.ndarray: The extracted HOG features.
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
            'orientations': self.orientations, 
            'pixels_per_cell': self.pixels_per_cell, 
            'cells_per_block': self.cells_per_block
        }
        save_artifact(state, path)

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic extractor.

        Args:
            path (str): The path to the saved state.
        """
        state = load_artifact(path)
            
        self.orientations = state['orientations']
        self.pixels_per_cell = state['pixels_per_cell']
        self.cells_per_block = state['cells_per_block']
        
        self.atomic_extractor = HOGExtractor(
            orientations=self.orientations, 
            pixels_per_cell=self.pixels_per_cell, 
            cells_per_block=self.cells_per_block
        )