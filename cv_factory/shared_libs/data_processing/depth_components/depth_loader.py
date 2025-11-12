# shared_libs/data_processing/depth_components/depth_loader.py
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_depth_processor import BaseDepthProcessor

logger = logging.getLogger(__name__)

# Giả định có thư viện đọc ảnh (cv2/PIL) đã được import và sử dụng
# import cv2 # Thư viện này cần được thêm vào khi tích hợp thực tế

class DepthLoader(BaseDepthProcessor):
    """
    Component chịu trách nhiệm tải, tiền xử lý cơ bản (đọc) và căn chỉnh 
    (alignment) cặp ảnh RGB và Depth Map từ đường dẫn file.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.depth_format = self.config.get("format", "png") # ví dụ: png, exr
        logger.info(f"DepthLoader initialized with format: {self.depth_format}")

    def _mock_load(self, path: str) -> np.ndarray:
        """Mô phỏng quá trình tải ảnh."""
        # Trong thực tế, đây là nơi gọi cv2.imread(path, ...)
        if 'rgb' in path.lower():
            # (H, W, 3) giả định
            return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
        else: # depth
            # (H, W) hoặc (H, W, 1) giả định
            return np.random.rand(480, 640).astype(np.float32) * 100.0 

    def process(self, rgb_path: str, depth_path: str) -> Dict[str, Any]:
        # Tải dữ liệu (Mô phỏng)
        rgb_image = self._mock_load(rgb_path)
        depth_map = self._mock_load(depth_path)
        
        # HARDENING: Check RGB/Depth size match
        if rgb_image.shape[:2] != depth_map.shape[:2]:
            raise ValueError(
                f"DepthLoader: RGB size {rgb_image.shape[:2]} does not match Depth size {depth_map.shape[:2]}. "
                "Requires alignment/resizing step."
            )
        
        logger.debug(f"Loaded and aligned RGB {rgb_image.shape} and Depth {depth_map.shape}")
        
        return {"rgb": rgb_image, "depth": depth_map}