# shared_libs/ml_core/pipeline_components_cv/depth/cv_depth_cleaner.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base_domain_adapter import BaseDomainAdapter

# Import Orchestrator thực tế từ data_processing/
from shared_libs.data_processing.depth_components.depth_processing_orchestrator import DepthProcessingOrchestrator

logger = logging.getLogger(__name__)

class CVDepthCleaner(BaseDomainAdapter):
    """
    Adapter component for Depth Cleaning (Load, Normalize, Validate).

    It wraps the DepthProcessingOrchestrator and provides the MLOps interface.
    This component is typically Stateful (Persistence) as its underlying Orchestrator might be.
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the DepthProcessingOrchestrator.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        # Instantiate the Adaptee
        processor = DepthProcessingOrchestrator(config=config)
        
        super().__init__(
            processor=processor, 
            name="CVDepthCleaner", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: Dict[str, str], y: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Executes the Depth Processing flow (loading and cleaning RGB and Depth paths).
        
        Args:
            X (Dict[str, str]): Input data, containing paths (e.g., {'rgb_path': '...', 'depth_path': '...'}).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            Dict[str, np.ndarray]: Processed data ({'rgb': array, 'depth': array}).
            
        Raises:
            ValueError: If required paths are missing in X.
        """
        if 'rgb_path' not in X or 'depth_path' not in X:
            raise ValueError("CVDepthCleaner expects input dictionary with 'rgb_path' and 'depth_path'.")
            
        # Delegation: BaseDomainAdapter will look for self.processor.run(X, y) or self.processor.run(X)
        # Since REQUIRES_TARGET_DATA is False, BaseDomainAdapter should call self.processor.run(X)
        
        # NOTE: Vì BaseDomainAdapter không biết về kwargs (rgb_path, depth_path),
        # ta cần gọi trực tiếp processor.run(rgb_path=X['rgb_path'], depth_path=X['depth_path'])
        
        # Tuy nhiên, để tuân thủ BaseDomainAdapter, ta cần ủy quyền:
        # Nếu processor.run được override trong BaseDomainAdapter, nó sẽ gọi run(X).
        # Ta giữ lại logic gọi trực tiếp để đảm bảo luồng hoạt động chính xác cho Orchestrator này.
        return self.processor.run(rgb_path=X['rgb_path'], depth_path=X['depth_path'])