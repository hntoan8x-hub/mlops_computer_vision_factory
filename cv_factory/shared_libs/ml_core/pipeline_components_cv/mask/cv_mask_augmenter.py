# shared_libs/ml_core/pipeline_components_cv/mask/cv_mask_augmenter.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Augmenter thực tế từ data_processing/
from shared_libs.data_processing.mask_components.mask_augmenter import MaskAugmenter

logger = logging.getLogger(__name__)

MaskData = Dict[str, np.ndarray]

class CVMaskAugmenter(BaseDomainAdapter):
    """
    Adapter component for Mask Augmentation.
    
    It wraps the atomic MaskAugmenter and provides the MLOps interface.
    This component is Stateless (ML) but Stateful (Persistence).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the MaskAugmenter.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        processor = MaskAugmenter(config=config)
        super().__init__(
            processor=processor, 
            name="CVMaskAugmenter", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: MaskData, y: Optional[Any] = None) -> MaskData:
        """
        Executes Mask Augmentation on image arrays (RGB and Mask).
        
        Args:
            X (MaskData): Input data, containing {'rgb': array, 'mask': array}.
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            MaskData: The augmented data.
            
        Raises:
            ValueError: If 'rgb' or 'mask' keys are missing in X.
        """
        if 'rgb' not in X or 'mask' not in X:
            raise ValueError("CVMaskAugmenter expects input dictionary with 'rgb' and 'mask' arrays.")
            
        # Delegation to processor.process (requires kwargs)
        return self.processor.process(rgb_image=X['rgb'], mask_map=X['mask'])