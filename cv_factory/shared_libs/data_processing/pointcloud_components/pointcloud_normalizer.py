# shared_libs/data_processing/pointcloud_components/pointcloud_normalizer.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_pointcloud_processor import BasePointcloudProcessor, PointcloudData

logger = logging.getLogger(__name__)

class PointcloudNormalizer(BasePointcloudProcessor):
    """
    Component chuẩn hóa tọa độ Point Cloud (Centering, Scaling) và 
    chuẩn hóa các thuộc tính (Intensity).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.normalize_method = self.config.get("method", "center_unit_sphere")
        self.normalize_intensity = self.config.get("normalize_intensity", True)
        logger.info(f"PC Normalizer initialized. Method: {self.normalize_method}")

    def process(self, pointcloud: PointcloudData) -> PointcloudData:
        
        current_pc = pointcloud.copy()
        coords = current_pc[:, :3] # x, y, z
        
        # 1. Centering (Trừ đi tâm)
        center = np.mean(coords, axis=0)
        coords -= center
        logger.debug(f"Centered Point Cloud (center: {center}).")
        
        # 2. Scaling (Đưa về hình cầu đơn vị)
        if self.normalize_method == "center_unit_sphere":
            # Tìm khoảng cách lớn nhất từ tâm
            max_distance = np.max(np.linalg.norm(coords, axis=1))
            if max_distance > 1e-6:
                coords /= max_distance
            logger.debug("Scaled Point Cloud to unit sphere.")
            
        current_pc[:, :3] = coords
        
        # 3. Chuẩn hóa Intensity (thuộc tính thứ 4)
        if self.normalize_intensity and current_pc.shape[1] > 3:
            intensity = current_pc[:, 3]
            min_val = np.min(intensity)
            max_val = np.max(intensity)
            
            if max_val - min_val > 1e-6:
                current_pc[:, 3] = (intensity - min_val) / (max_val - min_val)
            else:
                current_pc[:, 3] = 0.0 # Nếu tất cả bằng nhau
            logger.debug("Normalized Intensity channel.")

        # HARDENING: Xử lý NaN/Inf (nếu có)
        if not np.isfinite(current_pc).all():
            logger.warning("Point Cloud contains non-finite values (NaN/Inf). Replacing with zero.")
            current_pc = np.nan_to_num(current_pc)
            
        return current_pc