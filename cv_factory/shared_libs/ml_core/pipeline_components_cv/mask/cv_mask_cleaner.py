# shared_libs/ml_core/pipeline_components_cv/mask/cv_mask_cleaner.py (FIXED)

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from shared_libs.ml_core.pipeline_components_cv.base.base_domain_adapter import BaseDomainAdapter

# Import Orchestrator thực tế
from shared_libs.data_processing.mask_components.mask_processing_orchestrator import MaskProcessingOrchestrator

logger = logging.getLogger(__name__)

class CVMaskCleaner(BaseDomainAdapter):
    """
    Adapter component for Mask Cleaning (Load, Normalize, Validate).

    It wraps the MaskProcessingOrchestrator and provides the MLOps interface.
    This component is typically Stateful (Persistence).
    """
    
    # Inherits REQUIRES_TARGET_DATA: False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Adapter and the MaskProcessingOrchestrator.

        Args:
            config (Optional[Dict[str, Any]]): Component configuration.
        """
        processor = MaskProcessingOrchestrator(config=config)
        super().__init__(
            processor=processor, 
            name="CVMaskCleaner", 
            config=config
        )

    # FIX: Tuân thủ Signature Base bằng cách thêm y
    def transform(self, X: Dict[str, str], y: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Executes the Mask Processing flow (loading and cleaning RGB and Mask paths).
        
        Args:
            X (Dict[str, str]): Input data, containing paths (e.g., {'rgb_path': '...', 'mask_path': '...'}).
            y (Optional[Any]): Target data (ignored, as REQUIRES_TARGET_DATA is False).
                                
        Returns:
            Dict[str, np.ndarray]: Processed data ({'rgb': array, 'mask': array}).
            
        Raises:
            ValueError: If required paths are missing in X.
        """
        if 'rgb_path' not in X or 'mask_path' not in X:
            raise ValueError("CVMaskCleaner expects input dictionary with 'rgb_path' and 'mask_path'.")
            
        # Delegation to processor.run (requires kwargs)
        return self.processor.run(rgb_path=X['rgb_path'], mask_path=X['mask_path'])