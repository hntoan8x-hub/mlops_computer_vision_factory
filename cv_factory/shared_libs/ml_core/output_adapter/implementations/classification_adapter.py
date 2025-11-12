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
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: ClassificationAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        
        logits = self._to_numpy(raw_output)
        
        if logits.ndim == 1:
            logits = np.expand_dims(logits, axis=0)
            
        if logits.ndim != 2:
            raise ValueError(f"Classification output must be 2D ([B, C]), received shape: {logits.shape}")

        if self.params.is_logits:
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)
            probabilities = softmax(logits)
        else:
            probabilities = logits
        
        return probabilities

    def adapt_targets(self, targets_tensor: torch.Tensor) -> np.ndarray:
        """
        Chuẩn hóa Ground Truth Classification Targets cho Metric.update().
        
        Args:
            targets_tensor (torch.Tensor): Tensor Ground Truth thô (thường là LongTensor).
            
        Returns:
            np.ndarray: NumPy array chứa chỉ số lớp (int64).
        """
        # Chuyển đổi sang NumPy và đảm bảo kiểu dữ liệu là int64 cho chỉ số lớp
        targets_np = self._to_numpy(targets_tensor)
        
        # Nếu targets là List/Tuple (ví dụ: các nhãn không đồng nhất), cần xử lý thêm.
        # Giả định targets_tensor là LongTensor/IntTensor
        return targets_np.astype(np.int64)