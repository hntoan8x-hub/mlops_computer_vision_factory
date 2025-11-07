# shared_libs/ml_core/output_adapter/implementations/embedding_adapter.py

import numpy as np
import torch
from typing import Dict, Any, Union, List
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput
from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, EmbeddingAdapterParams

logger = logging.getLogger(__name__)

class EmbeddingAdapter(BaseOutputAdapter):
    """
    Adapter cho Embedding Learning: Chuyển đổi đầu ra thô (feature vector) 
    sang NumPy array 1D chuẩn hóa, áp dụng L2 norm nếu cần.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: EmbeddingAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        
        feature_vector = self._to_numpy(raw_output)
        
        # Đảm bảo vector là 1D hoặc [1, D]
        if feature_vector.ndim == 2 and feature_vector.shape[0] == 1:
            feature_vector = feature_vector.flatten()
        elif feature_vector.ndim != 1:
             raise ValueError(f"Embedding output must be 1D or [1, D], received shape: {feature_vector.shape}")
             
        # Kiểm tra kích thước vector
        if feature_vector.shape[0] != self.params.embedding_dim:
             logger.warning(f"Embedding dimension mismatch: Expected {self.params.embedding_dim}, got {feature_vector.shape[0]}")

        # Áp dụng L2 Normalization (sử dụng cấu hình normalize_vector)
        if self.params.normalize_vector:
            norm = np.linalg.norm(feature_vector)
            if norm > 1e-6:
                feature_vector = feature_vector / norm
        
        # Kiểm tra config return_dict (từ OutputAdapterConfig cấp cao)
        if self.adapter_config.return_dict:
             return {'vector': feature_vector} # Trả về Dict
        else:
             return feature_vector # Trả về np.ndarray