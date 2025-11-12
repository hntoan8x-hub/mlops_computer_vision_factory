# shared_libs/data_processing/depth_components/depth_normalizer.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_depth_processor import BaseDepthProcessor

logger = logging.getLogger(__name__)

class DepthNormalizer(BaseDepthProcessor):
    """
    Component chuẩn hóa Depth Map (ví dụ: Min-Max Scaling, Z-score) 
    và xử lý các giá trị không hợp lệ (NaN/Inf).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.method = self.config.get("method", "minmax")
        self.scale_range = self.config.get("scale", [0.0, 1.0])
        logger.info(f"DepthNormalizer initialized with method: {self.method}, range: {self.scale_range}")

    def process(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        
        # 1. HARDENING: Auto-detect and handle invalid values (NaN, Inf)
        if not np.isfinite(depth_map).all():
            nan_count = np.sum(np.isnan(depth_map))
            inf_count = np.sum(np.isinf(depth_map))
            logger.warning(f"Depth map contains {nan_count} NaNs and {inf_count} Infs. Replacing with zero.")
            
            # Thay thế các giá trị không hợp lệ (NaN hoặc Inf) bằng 0 (hoặc giá trị trung bình/median)
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Áp dụng chuẩn hóa (Min-Max Scaling)
        if self.method == "minmax":
            min_val = np.min(depth_map)
            max_val = np.max(depth_map)
            
            if max_val - min_val > 1e-6:
                depth_map = (depth_map - min_val) / (max_val - min_val)
            else:
                depth_map[:] = 0.0 # Tránh chia cho 0 nếu ảnh phẳng
            
            # Scale đến range mong muốn (ví dụ: [0, 1])
            output_min, output_max = self.scale_range
            depth_map = depth_map * (output_max - output_min) + output_min
        
        # 3. HARDENING: Clamp depth to prevent overflow/underflow
        depth_map = np.clip(depth_map, self.scale_range[0], self.scale_range[1])
            
        logger.debug(f"Depth map normalized using {self.method} to range {self.scale_range}")
        
        return {"rgb": rgb_image, "depth": depth_map}