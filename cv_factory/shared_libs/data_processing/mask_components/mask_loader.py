# shared_libs/data_processing/mask_components/mask_loader.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_mask_processor import BaseMaskProcessor, MaskData

logger = logging.getLogger(__name__)

class MaskLoader(BaseMaskProcessor):
    """
    Component tải, tiền xử lý cơ bản và căn chỉnh (alignment) cặp ảnh RGB và Mask/Label Map 
    từ đường dẫn file, đảm bảo chúng có cùng kích thước.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mask_type = self.config.get("mask_type", "segmentation") # 'segmentation', 'bbox', 'keypoints'
        logger.info(f"MaskLoader initialized for type: {self.mask_type}")

    def _mock_load_image(self, path: str, is_mask: bool) -> np.ndarray:
        """Mô phỏng quá trình tải ảnh/mask và resize về kích thước chuẩn."""
        H, W = 256, 256
        if 'rgb' in path.lower() or not is_mask:
            # Giả lập tải ảnh RGB (H, W, 3)
            return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) 
        else:
            # Giả lập tải Segmentation Mask (H, W) với 5 lớp (0-4)
            return np.random.randint(0, 5, (H, W), dtype=np.int32) 

    def process(self, rgb_path: str, mask_path: str) -> Dict[str, Any]:
        """
        Tải và căn chỉnh cặp RGB/Mask.
        """
        # Tải dữ liệu (Mô phỏng)
        rgb_image = self._mock_load_image(rgb_path, is_mask=False)
        mask_map = self._mock_load_image(mask_path, is_mask=True)
        
        # HARDENING: Check RGB/Mask size match
        if rgb_image.shape[:2] != mask_map.shape[:2]:
            raise ValueError(
                f"MaskLoader: RGB size {rgb_image.shape[:2]} does not match Mask size {mask_map.shape[:2]}. "
            )
        
        # Nếu là Bbox/Keypoints, mask_map sẽ là Dict. Giữ nguyên là np.ndarray trong ví dụ này.
        if self.mask_type == "bbox":
            mask_data = {"boxes": np.array([[10, 10, 50, 50]]), "labels": np.array([1])}
        else:
            mask_data = mask_map
            
        logger.debug(f"Loaded and aligned RGB {rgb_image.shape} and Mask/Label Map {mask_map.shape}")
        
        return {"rgb": rgb_image, "mask": mask_data}