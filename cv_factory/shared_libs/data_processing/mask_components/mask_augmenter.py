# shared_libs/data_processing/mask_components/mask_augmenter.py
import logging
import numpy as np
import random
from typing import Dict, Any, Optional
from .base_mask_processor import BaseMaskProcessor, MaskData

logger = logging.getLogger(__name__)

class MaskAugmenter(BaseMaskProcessor):
    """
    Component áp dụng các kỹ thuật Augmentation đồng bộ (synchronized) 
    cho cả RGB Image và Mask/Label Map/BBox để đảm bảo tính nhất quán.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.apply_flip = self.config.get("apply_flip", False)
        self.apply_rotate = self.config.get("apply_rotate", False)
        logger.info(f"MaskAugmenter initialized. Flip: {self.apply_flip}, Rotate: {self.apply_rotate}")

    def _apply_flip(self, data: np.ndarray) -> np.ndarray:
        return np.flip(data, axis=1).copy()
    
    def _apply_rotate(self, data: np.ndarray, angle: int) -> np.ndarray:
        # NOTE: Trong thực tế, cần dùng thư viện (OpenCV/Pillow) để Rotate chính xác.
        # Ở đây chỉ mô phỏng phép chuyển vị (transpose) đơn giản cho 90/270 độ.
        if angle == 90:
            return np.rot90(data, k=1)
        elif angle == 270:
            return np.rot90(data, k=3)
        return data

    def process(self, rgb_image: np.ndarray, mask_data: MaskData) -> Dict[str, Any]:
        
        current_rgb = rgb_image
        current_mask = mask_data
        
        # 1. Đồng bộ Horizontal Flip
        if self.apply_flip and random.random() < 0.5:
            current_rgb = self._apply_flip(current_rgb)
            if isinstance(current_mask, np.ndarray):
                current_mask = self._apply_flip(current_mask)
            elif isinstance(current_mask, dict) and 'boxes' in current_mask:
                # HARDENING: Cần logic phức tạp để flip BBoxes
                W = current_rgb.shape[1]
                boxes = current_mask['boxes']
                x_min, y_min, x_max, y_max = boxes.T
                
                # Flip BBoxes: x_min -> W - x_max, x_max -> W - x_min
                current_mask['boxes'] = np.stack([W - x_max, y_min, W - x_min, y_max], axis=1)
                
            logger.debug("Applied Synchronized Horizontal Flip.")

        # 2. Đồng bộ Rotation (90 độ)
        if self.apply_rotate and random.random() < 0.25:
            angle = random.choice([90, 270])
            current_rgb = self._apply_rotate(current_rgb, angle)
            if isinstance(current_mask, np.ndarray):
                current_mask = self._apply_rotate(current_mask, angle)
            # Logic xoay BBoxes phức tạp hơn, ta bỏ qua trong mô phỏng này.
            
            logger.debug(f"Applied Synchronized Rotation {angle} degrees.")
        
        return {"rgb": current_rgb, "mask": current_mask}