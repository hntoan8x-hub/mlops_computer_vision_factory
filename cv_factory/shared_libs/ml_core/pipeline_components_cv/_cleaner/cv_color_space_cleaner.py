# shared_libs/ml_core/pipeline_components_cv/_cleaner/cv_color_space_cleaner.py (UPDATED)

import logging
import numpy as np
from typing import Dict, Any, Union, Optional, Literal, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.cleaners.atomic.color_space_cleaner import ColorSpaceCleaner # Atomic Logic

logger = logging.getLogger(__name__)

ColorSpaceConversion = Literal["BGR2RGB", "RGB2BGR", "BGR2GRAY", "RGB2GRAY"]

class CVColorSpaceCleaner(BaseComponent):
    """
    Adapter component for color space conversion. 
    
    Manages the conversion code configuration, implements persistence, and delegates 
    transformation to the atomic ColorSpaceCleaner. This component is Stateless (non-learning).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, conversion_code: ColorSpaceConversion):
        """
        Initializes the Adapter and the Atomic Cleaner.

        Args:
            conversion_code (ColorSpaceConversion): The code for color space conversion, e.g., "BGR2RGB".
        """
        self.conversion_code = conversion_code
        
        self.atomic_cleaner = ColorSpaceCleaner(conversion_code=self.conversion_code) 
        
        logger.info(f"Initialized CVColorSpaceCleaner Adapter with code: {self.conversion_code}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVColorSpaceCleaner':
        """
        CVColorSpaceCleaner is a stateless component, no fitting is required.

        Args:
            X (Any): Input data (ignored).
            y (Optional[Any]): Target data (ignored).

        Returns:
            CVColorSpaceCleaner: The component instance.
        """
        logger.info("CVColorSpaceCleaner is stateless, no fitting required.")
        return self

    # Sửa lỗi hợp đồng: Bắt buộc nhận y: Optional[Any] = None
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Applies color space conversion by delegating the execution to the atomic cleaner.
        
        Args:
            X (np.ndarray): The input image array(s).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).

        Returns:
            np.ndarray: The color-space converted image array(s).
        """
        return self.atomic_cleaner.transform(X)

    def save(self, path: str) -> None:
        """
        Saves the component's configurable state (its conversion code) for persistence.

        Args:
            path (str): The path to save the state dictionary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {'conversion_code': self.conversion_code}
        save_artifact(state, path)

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic cleaner.

        Args:
            path (str): The path to the saved state.
        """
        state = load_artifact(path)
            
        self.conversion_code = state['conversion_code']
        
        self.atomic_cleaner = ColorSpaceCleaner(conversion_code=self.conversion_code)