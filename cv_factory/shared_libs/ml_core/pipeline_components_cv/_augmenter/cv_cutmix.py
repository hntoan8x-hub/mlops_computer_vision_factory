# cv_factory/shared_libs/ml_core/pipeline_components_cv/_augmenter/cv_cutmix.py (FIXED)

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.augmenters.atomic.cutmix import CutMix 

logger = logging.getLogger(__name__)

class CVCutMix(BaseComponent):
    """
    Adapter component for applying CutMix augmentation.
    
    This technique mixes two samples and their labels (REQUIRES_TARGET_DATA=True).
    """
    
    # FIX: Thêm Hợp đồng Tĩnh BẮT BUỘC
    REQUIRES_TARGET_DATA: bool = True

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the Adapter and the Atomic CutMix logic.
        
        Args:
            alpha (float): Beta distribution parameter.
        """
        self.alpha = alpha
        self.atomic_augmenter = CutMix(alpha=self.alpha)
        logger.info("Initialized CVCutMix Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVCutMix':
        """CVCutMix is typically stateless, no fitting required."""
        logger.info("CVCutMix is typically stateless (applied dynamically), no fitting required.")
        return self

    # FIX: Signature đã nhận y, cần đảm bảo trả về Tuple (X', Y')
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Applies CutMix by delegating execution to the atomic augmenter.

        Args:
            X (np.ndarray): The batch of input images.
            y (Optional[Any]): The batch of target labels.

        Returns:
            Tuple[np.ndarray, Optional[Any]]: The mixed images (X') and mixed labels (Y').
        
        Raises:
            ValueError: If y is None, as CutMix requires labels.
        """
        if y is None:
            raise ValueError("CutMix requires labels (y) for label mixing.")
        
        # NOTE: Tương tự Mixup, giả định atomic_augmenter.transform(X, y) trả về (X_mixed, Y_mixed)
        return self.atomic_augmenter.transform(X, labels=y)

    def save(self, path: str) -> None:
        """Saves the component's state (alpha) using the utility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {'alpha': self.alpha}
        save_artifact(state, path)

    def load(self, path: str) -> None:
        """
        Loads the component's state and re-initializes the atomic augmenter.
        """
        state = load_artifact(path)
        self.alpha = state['alpha']
        self.atomic_augmenter = CutMix(alpha=self.alpha)