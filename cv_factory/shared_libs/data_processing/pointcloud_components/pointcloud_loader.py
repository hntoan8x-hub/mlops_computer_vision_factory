# shared_libs/data_processing/pointcloud_components/pointcloud_loader.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_pointcloud_processor import BasePointcloudProcessor, PointcloudData

logger = logging.getLogger(__name__)

class PointcloudLoader(BasePointcloudProcessor):
    """
    Component tải dữ liệu Point Cloud từ đường dẫn file (.pcd, .bin, .ply) 
    và chuyển đổi nó thành ma trận numpy (N, M).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.output_format = self.config.get("format", "xyz_i") # x, y, z, intensity
        self.max_points = self.config.get("max_points", 16384)
        logger.info(f"PointcloudLoader initialized. Max points: {self.max_points}")

    def _mock_load(self, path: str) -> PointcloudData:
        """Mô phỏng quá trình tải và trả về Point Cloud (N, 4)"""
        N = np.random.randint(self.max_points * 0.5, self.max_points)
        # Tạo dữ liệu x, y, z (từ 0 đến 100) và intensity (0 đến 1)
        data = np.random.rand(N, 4).astype(np.float32)
        data[:, :3] *= 100.0 
        return data

    def process(self, pc_path: str) -> PointcloudData:
        """
        Tải Point Cloud và áp dụng downsampling/padding.
        """
        pointcloud = self._mock_load(pc_path)
        N_original = pointcloud.shape[0]
        
        # HARDENING: Downsampling hoặc Padding để có số lượng điểm cố định
        if N_original > self.max_points:
            # Downsampling: Chọn ngẫu nhiên
            indices = np.random.choice(N_original, self.max_points, replace=False)
            pointcloud = pointcloud[indices, :]
            logger.debug(f"Downsampled from {N_original} to {self.max_points} points.")
        elif N_original < self.max_points:
            # Padding: Lặp lại hoặc thêm điểm 0
            padding_needed = self.max_points - N_original
            padding_data = np.zeros((padding_needed, pointcloud.shape[1]), dtype=pointcloud.dtype)
            pointcloud = np.concatenate([pointcloud, padding_data], axis=0)
            logger.debug(f"Padded from {N_original} to {self.max_points} points.")
            
        return pointcloud