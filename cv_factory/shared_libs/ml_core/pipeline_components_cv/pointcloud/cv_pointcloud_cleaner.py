# shared_libs/ml_core/pipeline_components_cv/pointcloud/cv_pointcloud_cleaner.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Orchestrator thực tế từ data_processing/
from shared_libs.data_processing.pointcloud_components.pointcloud_processing_orchestrator import PointcloudProcessingOrchestrator

logger = logging.getLogger(__name__)

PointCloudData = Union[str, np.ndarray] # File path or raw point array

class CVPointCloudCleaner(BaseDomainAdapter):
    """
    Adapter component for Point Cloud Cleaning (Load, Normalizer, Voxelizer).

    It wraps the PointcloudProcessingOrchestrator and provides the MLOps interface.
    This component is typically Stateful (Persistence).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the PointcloudProcessingOrchestrator.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        processor = PointcloudProcessingOrchestrator(config=config)
        super().__init__(
            processor=processor, 
            name="CVPointCloudCleaner", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: PointCloudData, y: Optional[Any] = None) -> np.ndarray:
        """
        Executes the Point Cloud Processing flow.
        
        Args:
            X (PointCloudData): Input data (File Path or raw Point Cloud array).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            np.ndarray: The processed Point Cloud (e.g., Voxelized Array or Normalized Points).
        """
        # Delegation to processor.run (requires data=X)
        return self.processor.run(data=X)