# cv_factory/shared_libs/ml_core/pipeline_components_cv/_augmenter/cv_mixup.py (FIXED)

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import os
from shared_libs.core_utils.io_utils import save_artifact, load_artifact
from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.data_processing.image_components.augmenters.atomic.mixup import Mixup 

logger = logging.getLogger(__name__)

class CVMixup(BaseComponent):
    """
    Adapter component for applying Mixup augmentation. 
    
    This technique mixes two samples and their labels (REQUIRES_TARGET_DATA=True).
    """
    
    # FIX: Thêm Hợp đồng Tĩnh BẮT BUỘC
    REQUIRES_TARGET_DATA: bool = True 

    def __init__(self, alpha: float = 0.2):
        """
        Initializes the Adapter and the Atomic Mixup logic.

        Args:
            alpha (float): Beta distribution parameter for controlling the mix ratio.
        """
        self.alpha = alpha
        self.atomic_augmenter = Mixup(alpha=self.alpha)
        logger.info("Initialized CVMixup Adapter.")

    def fit(self, X: Any, y: Optional[Any] = None) -> 'CVMixup':
        """
        CVMixup is typically stateless, no fitting required.
        """
        logger.info("CVMixup is typically stateless, no fitting required.")
        return self

    # FIX: Signature đã nhận y, cần đảm bảo trả về Tuple (X', Y')
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Applies Mixup by delegating execution to the atomic augmenter.

        Args:
            X (np.ndarray): The batch of input images.
            y (Optional[Any]): The batch of target labels.

        Returns:
            Tuple[np.ndarray, Optional[Any]]: The mixed images (X') and mixed labels (Y').
            
        Raises:
            ValueError: If y is None, as Mixup requires labels.
        """
        if y is None:
            # Check này vẫn cần thiết vì Orchestrator dựa vào REQUIRES_TARGET_DATA để truyền Y
            # và Adapter cần kiểm tra tính hợp lệ của Y.
            raise ValueError("Mixup requires labels (y) for label mixing.")
        
        # NOTE: Giả định atomic_augmenter.transform(X, y) trả về (X_mixed, Y_mixed) hoặc chỉ X_mixed
        # Chúng ta buộc phải giả định nó trả về (X_mixed, Y_mixed) để Orchestrator hoạt động.
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
        self.atomic_augmenter = Mixup(alpha=self.alpha)