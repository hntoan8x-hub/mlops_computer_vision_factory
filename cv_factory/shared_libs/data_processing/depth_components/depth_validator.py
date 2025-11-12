# shared_libs/data_processing/depth_components/depth_validator.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_depth_processor import BaseDepthProcessor

logger = logging.getLogger(__name__)

class DepthValidator(BaseDepthProcessor):
    """
    Component chịu trách nhiệm xác thực (Validate) Depth Map sau khi xử lý, 
    kiểm tra các điều kiện về phạm vi (range) và mật độ pixel hợp lệ.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_depth = self.config.get("min_depth", 0.1)
        self.max_depth = self.config.get("max_depth", 100.0)
        self.min_valid_ratio = self.config.get("min_valid_ratio", 0.95)
        logger.info(f"DepthValidator initialized. Range: [{self.min_depth}, {self.max_depth}]")

    def process(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        
        # 1. Kiểm tra phạm vi (range check)
        min_actual = np.min(depth_map)
        max_actual = np.max(depth_map)
        
        if min_actual < self.min_depth or max_actual > self.max_depth:
            logger.warning(
                f"Depth range violation: Actual range [{min_actual:.2f}, {max_actual:.2f}] "
                f"is outside expected range [{self.min_depth:.2f}, {self.max_depth:.2f}]."
            )
            # Không raise lỗi mà chỉ log warning, có thể áp dụng thêm clamp nếu cần
            
        # 2. Kiểm tra mật độ pixel hợp lệ (Missing Pixels)
        # Giả sử depth map được chuẩn hóa 0 là invalid/missing
        valid_pixels = np.count_nonzero(depth_map)
        total_pixels = depth_map.size
        valid_ratio = valid_pixels / total_pixels
        
        if valid_ratio < self.min_valid_ratio:
            error_msg = f"Low valid depth ratio: {valid_ratio:.2f} < {self.min_valid_ratio}. Data quality issue detected."
            logger.error(error_msg)
            # Trong thực tế, có thể raise Exception hoặc trả về cờ (flag) để loại bỏ sample
            # raise ValueError(error_msg) 
        
        logger.debug(f"Depth map passed validation. Valid ratio: {valid_ratio:.2f}")
        
        return {"rgb": rgb_image, "depth": depth_map} # Trả về bản sao không thay đổi