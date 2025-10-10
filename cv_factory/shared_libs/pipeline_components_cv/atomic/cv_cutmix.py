# cv_factory/shared_libs/ml_core/pipeline_components_cv/atomic/cv_cutmix.py

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import pickle

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.augmenters.atomic.cutmix import CutMix 

logger = logging.getLogger(__name__)

class CVCutMix(BaseComponent):
    """
    Adapter component for applying CutMix augmentation.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the Adapter and the Atomic CutMix logic.
        """
        self.alpha = alpha
        
        # Instantiate the Atomic Logic Class (The Adaptee)
        self.atomic_augmenter = CutMix(alpha=self.alpha)
        
        logger.info("Initialized CVCutMix Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVCutMix':
        logger.info("CVCutMix is typically stateless (applied dynamically), no fitting required.")
        return self

    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Applies CutMix by delegating execution to the atomic augmenter.
        CutMix often requires both X (image) and y (label).
        """
        # <<< ADAPTER LOGIC: Delegation of transformation >>>
        # Assuming the atomic layer has a method that handles both image and label mixing
        return self.atomic_augmenter.transform(X, y)
        # End of Adapter Logic

    # save/load methods only manage the alpha parameter, following the same pattern as FlipRotate