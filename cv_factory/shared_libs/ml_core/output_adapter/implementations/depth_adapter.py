# shared_libs/ml_core/output_adapter/implementations/depth_adapter.py

import numpy as np
import torch
from typing import Dict, Any, Union, List
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput
# Import schemas mới
from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, DepthAdapterParams

logger = logging.getLogger(__name__)

class DepthAdapter(BaseOutputAdapter):
    """
    Adapter cho Depth Estimation: Chuyển đổi đầu ra thô (Depth Map Tensor) 
    sang NumPy array float, áp dụng lọc ngưỡng min/max và squeeze chiều kênh.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: DepthAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        """
        Chuẩn hóa đầu ra dự đoán Depth Map (Predictions).
        """
        # 1. Chuyển đổi sang NumPy (Float)
        # Giả định raw_output là tensor depth map (B, 1, H, W)
        depth_map = self._to_numpy(raw_output).astype(np.float32)
        
        # 2. Squeeze chiều kênh (nếu cấu hình)
        if self.params.squeeze_channel and depth_map.ndim == 4 and depth_map.shape[1] == 1:
            depth_map = np.squeeze(depth_map, axis=1) # [B, 1, H, W] -> [B, H, W]

        # 3. Áp dụng Lọc Ngưỡng (Clipping)
        # Hữu ích cho các mô hình Regression để tránh giá trị NaN/inf hoặc outlier cực đoan
        depth_map = np.clip(depth_map, self.params.min_depth, self.params.max_depth)
        
        # 4. Trả về định dạng chuẩn hóa
        if self.adapter_config.return_dict:
             return {'depth_map': depth_map} 
        else:
             # Trả về NumPy array (B, H, W) hoặc (B, 1, H, W)
             return depth_map 

    def adapt_targets(self, targets_tensor: torch.Tensor) -> np.ndarray:
        """
        Chuẩn hóa Ground Truth Depth Map (Targets) cho Metric.update().
        
        Args:
            targets_tensor (torch.Tensor): Tensor Ground Truth thô.
            
        Returns:
            np.ndarray: NumPy array float chuẩn hóa.
        """
        # Targets phải là float type để Metric tính toán (regression)
        depth_map_gt = targets_tensor.float()
        
        # 1. Chuyển đổi sang NumPy
        depth_map_gt_np = self._to_numpy(depth_map_gt)
        
        # 2. Squeeze chiều kênh (để nhất quán với output của model)
        if self.params.squeeze_channel and depth_map_gt_np.ndim == 4 and depth_map_gt_np.shape[1] == 1:
             depth_map_gt_np = np.squeeze(depth_map_gt_np, axis=1)
        
        return depth_map_gt_np