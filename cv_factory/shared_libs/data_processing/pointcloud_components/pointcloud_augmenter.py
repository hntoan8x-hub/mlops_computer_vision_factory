# shared_libs/data_processing/pointcloud_components/pointcloud_augmenter.py
import logging
import numpy as np
import random
from typing import Dict, Any, Optional
from .base_pointcloud_processor import BasePointcloudProcessor, PointcloudData

logger = logging.getLogger(__name__)

class PointcloudAugmenter(BasePointcloudProcessor):
    """
    Component áp dụng các kỹ thuật Augmentation 3D: Rotation, Jittering, Scaling.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.apply_rotation = self.config.get("apply_rotation", True)
        self.jitter_std = self.config.get("jitter_std", 0.01)
        self.scale_range = self.config.get("scale_range", [0.95, 1.05])
        logger.info(f"PC Augmenter initialized. Jitter: {self.jitter_std}")

    def _get_rotation_matrix(self, axis: str, angle: float) -> np.ndarray:
        """Tạo ma trận xoay 3x3 đơn giản."""
        cos = np.cos(angle)
        sin = np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
        if axis == 'y':
            return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        if axis == 'z':
            return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        return np.identity(3)

    def process(self, pointcloud: PointcloudData) -> PointcloudData:
        
        current_pc = pointcloud.copy()
        coords = current_pc[:, :3]
        
        # 1. Rotation (thường là quay quanh trục Z)
        if self.apply_rotation and random.random() < 0.75:
            angle_z = random.uniform(0, 2 * np.pi)
            R_z = self._get_rotation_matrix('z', angle_z)
            coords = np.dot(coords, R_z.T)
            logger.debug("Applied random rotation around Z-axis.")

        # 2. Scaling (Áp dụng scale đồng bộ)
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        coords *= scale_factor
        logger.debug(f"Applied scale factor: {scale_factor:.2f}.")

        # 3. Jittering (Thêm nhiễu Gaussian nhỏ)
        if self.jitter_std > 0:
            jitter = np.random.normal(0, self.jitter_std, coords.shape).astype(coords.dtype)
            coords += jitter
            logger.debug(f"Applied jittering (std: {self.jitter_std}).")

        current_pc[:, :3] = coords
        return current_pc