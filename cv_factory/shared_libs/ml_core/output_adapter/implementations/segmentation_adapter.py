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
            predicted_indices = np.argmax(logits_or_masks, axis=1)
            
        elif logits_or_masks.ndim == 3:
            predicted_indices = logits_or_masks
            
        else:
            raise ValueError("Segmentation Adapter only supports 4D logits or 3D indices/masks.")

        predicted_indices = predicted_indices.astype(np.int64)

        return predicted_indices

    def adapt_targets(self, targets_tensor: torch.Tensor) -> np.ndarray:
        """
        Chuẩn hóa Ground Truth Segmentation Targets cho Metric.update().
        
        Args:
            targets_tensor (torch.Tensor): Tensor Ground Truth thô (thường là LongTensor [B, H, W]).
            
        Returns:
            np.ndarray: NumPy array chứa chỉ số lớp (int64).
        """
        # Targets phải là LongTensor chứa chỉ số lớp.
        targets_np = self._to_numpy(targets_tensor)
        return targets_np.astype(np.int64)