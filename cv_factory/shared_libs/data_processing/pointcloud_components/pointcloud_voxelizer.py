# shared_libs/data_processing/pointcloud_components/pointcloud_voxelizer.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_pointcloud_processor import BasePointcloudProcessor, PointcloudData

logger = logging.getLogger(__name__)

# Voxelization là một kỹ thuật chuyển đổi Point Cloud (rời rạc) sang Voxel Grid (lưới 3D)

class PointcloudVoxelizer(BasePointcloudProcessor):
    """
    Component chuyển đổi dữ liệu Point Cloud (N, M) thành định dạng Voxel Grid (L, W, H, C) 
    để sử dụng cho các mô hình 3D Convolution.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.voxel_size = self.config.get("voxel_size", 0.1) # Kích thước 1 voxel (mét)
        self.grid_shape = self.config.get("grid_shape", [32, 32, 32])
        self.enabled = self.config.get("enabled", False)
        logger.info(f"PC Voxelizer initialized. Grid shape: {self.grid_shape}")

    def _mock_voxelize(self, coords: np.ndarray, grid_shape: List[int]) -> np.ndarray:
        """Mô phỏng quá trình Voxelization: Trả về một mảng 4D chứa mật độ điểm."""
        # Giả định PC đã được chuẩn hóa về phạm vi [-1, 1] hoặc [0, 1]
        
        # Voxelization rất chậm, chỉ trả về một mảng giả lập
        L, W, H = grid_shape
        # C là số kênh (ví dụ: mật độ, intensity trung bình, normal vector)
        C = coords.shape[1] if coords.shape[1] > 3 else 1 
        
        voxel_grid = np.random.rand(L, W, H, C).astype(np.float32)
        
        # HARDENING: Đảm bảo có một số giá trị > 0
        voxel_grid[voxel_grid < 0.95] = 0.0
        
        return voxel_grid

    def process(self, pointcloud: PointcloudData) -> PointcloudData:
        """
        Thực hiện Voxelization nếu được bật.
        """
        if not self.enabled:
            # Nếu không bật Voxelization, trả về Point Cloud gốc
            return pointcloud

        coords = pointcloud[:, :3]
        
        # Voxelization (Mô phỏng)
        voxel_grid = self._mock_voxelize(coords, self.grid_shape)
        
        logger.debug(f"Point Cloud Voxelized to grid shape: {voxel_grid.shape}")
        
        # Voxelizer thường là bước cuối cùng của PC Processing, trả về 4D Array
        return voxel_grid