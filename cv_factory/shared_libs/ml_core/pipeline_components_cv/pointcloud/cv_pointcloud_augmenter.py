# shared_libs/ml_core/pipeline_components_cv/pointcloud/cv_pointcloud_augmenter.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Augmenter thực tế từ data_processing/
from shared_libs.data_processing.pointcloud_components.pointcloud_augmenter import PointcloudAugmenter

logger = logging.getLogger(__name__)

class CVPointCloudAugmenter(BaseDomainAdapter):
    """
    Adapter component for Point Cloud Augmentation (Rotation, Jittering).

    It wraps the atomic PointcloudAugmenter and provides the MLOps interface.
    This component is Stateless (ML) but Stateful (Persistence).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the PointcloudAugmenter.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        processor = PointcloudAugmenter(config=config)
        super().__init__(
            processor=processor, 
            name="CVPointCloudAugmenter", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Executes Point Cloud Augmentation.
        
        Args:
            X (np.ndarray): Input Point Cloud array (Nx3 or NxM).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            np.ndarray: The augmented Point Cloud.
        """
        # Delegation to processor.process (requires data=X)
        return self.processor.process(data=X)