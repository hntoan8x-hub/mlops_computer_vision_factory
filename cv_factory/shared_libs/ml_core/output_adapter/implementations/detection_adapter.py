# shared_libs/ml_core/output_adapter/implementations/detection_adapter.py (UPDATED)

import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput
# IMPORT CONFIG MỚI
from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, DetectionAdapterParams

logger = logging.getLogger(__name__)

class DetectionAdapter(BaseOutputAdapter):
    """
    Adapter cho Object Detection: Chuyển đổi đầu ra thô sang định dạng List[Dict] 
    chuẩn hóa, áp dụng chuẩn hóa BBox nếu cần.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: DetectionAdapterParams = self.adapter_config.params
        
    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        
        predictions_list: List[Dict[str, Any]] = []
        image_size = kwargs.get('image_size') # Lấy từ kwargs trong CVDataset/Predictor

        # Giả định đầu ra là List[Dict] chuẩn hóa theo mẫu của PyTorch/Torchvision
        if not isinstance(raw_output, list) or not all(isinstance(item, dict) for item in raw_output):
             # Nếu không phải List[Dict], cần logic phức tạp hơn để xử lý tensor thô
             raise TypeError("Detection Adapter expects raw_output as List[Dict] of Tensors.")

        for item in raw_output:
            boxes = self._to_numpy(item[self.params.box_key]) 
            scores = self._to_numpy(item['scores']) 
            labels = self._to_numpy(item['labels'])
            
            # Xử lý BBox
            for box, score, label in zip(boxes, scores, labels):
                final_box = box
                
                # Áp dụng chuẩn hóa nếu cần và nếu chưa chuẩn hóa
                if self.params.normalize and image_size and box.max() > 1.0:
                    H, W = image_size
                    # Giả sử box là [xmin, ymin, xmax, ymax]
                    final_box = [box[0] / W, box[1] / H, box[2] / W, box[3] / H]
                    
                # NOTE: Cần thêm logic xử lý input_bbox_format (xywh -> xyxy) nếu cần

                predictions_list.append({
                    'box': np.array(final_box),  # np.ndarray (xyxy normalized)
                    'score': float(score),       # float
                    'class': int(label)          # int
                })

        return predictions_list