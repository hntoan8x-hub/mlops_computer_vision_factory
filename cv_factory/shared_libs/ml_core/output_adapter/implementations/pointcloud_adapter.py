# shared_libs/ml_core/output_adapter/implementations/pointcloud_adapter.py (NEW)

import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple, Optional
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput
from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, PointCloudAdapterParams

logger = logging.getLogger(__name__)

class PointCloudAdapter(BaseOutputAdapter):
    """
    Adapter cho Point Cloud Processing: Chuẩn hóa đầu ra 3D (Segmentation Masks, 3D BBoxes).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: PointCloudAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        """
        Chuẩn hóa đầu ra dự đoán Point Cloud.
        """
        output_data = self._to_numpy(raw_output)
        
        # Xử lý 3D Detection (trả về List[Dict])
        if self.params.box_3d_key and isinstance(output_data, dict):
            
            boxes = output_data.get(self.params.box_3d_key)
            if boxes is None:
                 raise KeyError(f"Missing expected 3D box key: {self.params.box_3d_key}")
            
            # Giả định boxes là [N, 7] (x, y, z, l, w, h, yaw)
            standardized_predictions = []
            for box in boxes:
                # NOTE: Cần thêm logic normalize_coordinates nếu True
                standardized_predictions.append({
                    'box_3d': box,
                    'type': 'detection',
                    # ... (Thêm scores/labels nếu có)
                })
            return standardized_predictions
            
        # Xử lý 3D Segmentation (trả về NumPy array)
        elif self.params.segmentation_channel is not None and output_data.ndim > 1:
            # Giả định Output là [B, N_Points, N_Classes] hoặc [B, N_Classes, N_Points]
            
            if output_data.ndim == 3:
                # Lấy chỉ số lớp (index of max probability)
                predicted_indices = np.argmax(output_data, axis=self.params.segmentation_channel)
                return predicted_indices.astype(np.int64)
            else:
                return output_data
        
        else:
            # Trả về NumPy array thô (ví dụ: feature points)
            return output_data

    # Targets (Nhãn Point Cloud) thường là NumPy Array nên không cần xử lý Target phức tạp.
    def adapt_targets(self, targets_tensor: torch.Tensor) -> np.ndarray:
        """
        Chuẩn hóa Ground Truth Point Cloud Targets cho Metric.update().
        """
        # Targets thường là LongTensor chứa chỉ số lớp (Segmentation) hoặc Dict (Detection)
        return self._to_numpy(targets_tensor)