# shared_libs/data_processing/mask_components/mask_normalizer.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_mask_processor import BaseMaskProcessor, MaskData

logger = logging.getLogger(__name__)

class MaskNormalizer(BaseMaskProcessor):
    """
    Component chuẩn hóa Mask/Label Map sang định dạng phù hợp cho mô hình 
    (ví dụ: One-Hot Encoding cho Segmentation, hoặc chuẩn hóa tọa độ cho BBox).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.output_format = self.config.get("format", "long_label") # 'long_label', 'one_hot', 'normalized_bbox'
        self.num_classes = self.config.get("num_classes", 5)
        logger.info(f"MaskNormalizer initialized with output format: {self.output_format}")

    def process(self, rgb_image: np.ndarray, mask_data: MaskData) -> Dict[str, Any]:
        
        current_mask = mask_data
        
        # 1. Xử lý Segmentation Mask
        if isinstance(current_mask, np.ndarray):
            
            # HARDENING: Kiểm tra dtype, ép về int32
            if current_mask.dtype != np.int32:
                 current_mask = current_mask.astype(np.int32)

            if self.output_format == "one_hot":
                # Chuyển đổi sang One-Hot Encoding: (H, W) -> (H, W, C)
                if current_mask.ndim == 2:
                    H, W = current_mask.shape
                    one_hot_mask = np.zeros((H, W, self.num_classes), dtype=np.float32)
                    
                    # Áp dụng One-Hot
                    for c in range(self.num_classes):
                        one_hot_mask[current_mask == c, c] = 1.0
                        
                    current_mask = one_hot_mask
                
            logger.debug(f"Mask normalized to format '{self.output_format}'. Shape: {current_mask.shape}")

        # 2. Xử lý Bounding Boxes (Nếu mask_data là Dict)
        elif isinstance(current_mask, dict) and 'boxes' in current_mask:
            # Ví dụ: Chuẩn hóa tọa độ BBox từ pixel sang 0-1
            H, W = rgb_image.shape[:2]
            boxes = current_mask['boxes'] # [N, 4] (x_min, y_min, x_max, y_max)
            
            if self.output_format == "normalized_bbox":
                boxes = boxes.astype(np.float32)
                boxes[:, [0, 2]] /= W  # Normalize X
                boxes[:, [1, 3]] /= H  # Normalize Y
                current_mask['boxes'] = boxes
                logger.debug("BBoxes normalized to 0-1 range.")

        return {"rgb": rgb_image, "mask": current_mask}