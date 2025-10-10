# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_color_space_cleaner.py

import logging
import numpy as np
from typing import Dict, Any, Union, Optional, Literal
import pickle

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
# <<< CRITICAL: Import the Atomic Logic Class (the Adaptee) >>>
from shared_libs.data_processing.cleaners.atomic.color_space_cleaner import ColorSpaceCleaner 
# We assume ColorSpaceCleaner is the class provided in the prompt.

logger = logging.getLogger(__name__)

# Use Literal for type safety of color space conversions
ColorSpaceConversion = Literal["BGR2RGB", "RGB2BGR", "BGR2GRAY", "RGB2GRAY"]

class CVColorSpaceCleaner(BaseComponent):
    """
    Adapter component for color space conversion. 
    
    It adheres to the BaseComponent contract, manages the conversion code state, 
    and delegates the transformation to the atomic ColorSpaceCleaner class.
    """
    
    def __init__(self, conversion_code: ColorSpaceConversion):
        """
        Initializes the Adapter and the Atomic Cleaner.

        Args:
            conversion_code (ColorSpaceConversion): The code for color space conversion, e.g., "BGR2RGB".
        """
        # 1. Manage State/Parameters (Responsibility of the Adapter)
        self.conversion_code = conversion_code
        
        # 2. Instantiate the Atomic Logic Class (The Adaptee)
        # We instantiate the logic here, using the parameters managed by this Adapter
        self.atomic_cleaner = ColorSpaceCleaner(conversion_code=self.conversion_code) 
        
        logger.info(f"Initialized CVColorSpaceCleaner Adapter with code: {self.conversion_code}.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVColorSpaceCleaner':
        """
        CVColorSpaceCleaner is a stateless component, no fitting is required.
        """
        logger.info("CVColorSpaceCleaner is stateless, no fitting required.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies color space conversion by delegating the execution to the atomic cleaner.
        
        Args:
            X (np.ndarray): The input image array(s).

        Returns:
            np.ndarray: The color-space converted image array(s).
        """
        # <<< ADAPTER LOGIC: Delegation to the Atomic Layer >>>
        return self.atomic_cleaner.transform(X)
        # End of Adapter Logic

    def save(self, path: str) -> None:
        """
        Saves the component's state (its conversion code).
        """
        state = {'conversion_code': self.conversion_code}
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"CVColorSpaceCleaner state saved to {path}.")

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic cleaner.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.conversion_code = state['conversion_code']
        
        # Re-initialize the atomic logic with the loaded state
        self.atomic_cleaner = ColorSpaceCleaner(conversion_code=self.conversion_code) 
        logger.info(f"CVColorSpaceCleaner state loaded from {path}.")