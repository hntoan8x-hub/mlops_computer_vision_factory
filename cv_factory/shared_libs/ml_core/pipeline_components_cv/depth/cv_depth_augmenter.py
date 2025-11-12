# shared_libs/ml_core/pipeline_components_cv/depth/cv_depth_augmenter.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Augmenter thực tế từ data_processing/
from shared_libs.data_processing.depth_components.depth_augmenter import DepthAugmenter

logger = logging.getLogger(__name__)

# Depth Augmenter thường hoạt động trên mảng NumPy, không phải paths
DepthData = Dict[str, np.ndarray]

class CVDepthAugmenter(BaseDomainAdapter):
    """
    Adapter component for Depth Augmentation.
    
    It wraps the atomic DepthAugmenter and provides the MLOps interface.
    This component is Stateless (non-learning) but Stateful (Persistence).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the DepthAugmenter.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        # Instantiate the Adaptee
        processor = DepthAugmenter(config=config)
        super().__init__(
            processor=processor, 
            name="CVDepthAugmenter", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: DepthData, y: Optional[Any] = None) -> DepthData:
        """
        Executes Depth Augmentation on image arrays (RGB and Depth).
        
        Args:
            X (DepthData): Input data, containing {'rgb': array, 'depth': array}.
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            DepthData: The augmented data.
            
        Raises:
            ValueError: If 'rgb' or 'depth' keys are missing in X.
        """
        if 'rgb' not in X or 'depth' not in X:
            raise ValueError("CVDepthAugmenter expects input dictionary with 'rgb' and 'depth' arrays.")
            
        # Delegation: processor.process(rgb_image, depth_map)
        return self.processor.process(rgb_image=X['rgb'], depth_map=X['depth'])