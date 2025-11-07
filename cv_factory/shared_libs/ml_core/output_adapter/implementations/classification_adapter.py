# shared_libs/ml_core/output_adapter/implementations/classification_adapter.py (UPDATED)

import numpy as np
import torch
from typing import Dict, Any, Union, List
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput
# IMPORT CONFIG MỚI
from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, ClassificationAdapterParams

logger = logging.getLogger(__name__)

class ClassificationAdapter(BaseOutputAdapter):
    """
    Adapter cho Classification: Chuyển đổi Logits/Scores thô thành NumPy array chứa 
    probabilities, sử dụng cấu hình từ schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Validate và lấy params cụ thể
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: ClassificationAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        
        logits = self._to_numpy(raw_output)
        
        if logits.ndim == 1:
            logits = np.expand_dims(logits, axis=0)
            
        if logits.ndim != 2:
            raise ValueError(f"Classification output must be 2D ([B, C]), received shape: {logits.shape}")

        # Sử dụng cấu hình is_logits
        if self.params.is_logits:
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)
            probabilities = softmax(logits)
        else:
            probabilities = logits

        # NOTE: Logic lọc theo confidence_threshold có thể được áp dụng ở đây
        # hoặc thường được đẩy sang Domain Postprocessor/Metric.
        
        return probabilities