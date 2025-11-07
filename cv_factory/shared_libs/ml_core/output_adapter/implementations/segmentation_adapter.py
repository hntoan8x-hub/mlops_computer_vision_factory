# shared_libs/ml_core/output_adapter/implementations/segmentation_adapter.py (UPDATED)

import numpy as np
import torch
from typing import Dict, Any, Union, List
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput
# IMPORT CONFIG MỚI
from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, SegmentationAdapterParams

logger = logging.getLogger(__name__)

class SegmentationAdapter(BaseOutputAdapter):
    """
    Adapter cho Segmentation: Chuyển đổi Logits/Masks thô thành NumPy array chứa 
    chỉ số lớp dự đoán (class indices).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: SegmentationAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        
        logits_or_masks = self._to_numpy(raw_output)
        
        if logits_or_masks.ndim not in [3, 4]:
            raise ValueError(f"Segmentation output shape is unsupported: {logits_or_masks.shape}")

        if logits_or_masks.ndim == 4 and self.params.is_logits:
            # Format: [B, C, H, W] -> Logits PyTorch. Áp dụng argmax.
            predicted_indices = np.argmax(logits_or_masks, axis=1) # Kết quả: [B, H, W]
            
        elif logits_or_masks.ndim == 3:
            # Format: [B, H, W] (Đã là chỉ số lớp hoặc mask binary 3D)
            predicted_indices = logits_or_masks
            
        else:
            # Format 4D nhưng không phải logits (ví dụ: one-hot)
            raise ValueError("Segmentation Adapter only supports 4D logits or 3D indices/masks.")

        # Đảm bảo dtype là integer (class index)
        predicted_indices = predicted_indices.astype(np.int64)

        return predicted_indices