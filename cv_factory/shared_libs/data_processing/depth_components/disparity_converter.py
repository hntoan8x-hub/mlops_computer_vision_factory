# shared_libs/data_processing/depth_components/disparity_converter.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_depth_processor import BaseDepthProcessor

logger = logging.getLogger(__name__)

class DisparityConverter(BaseDepthProcessor):
    """
    Component chuyển đổi giữa Depth Map (Z) và Disparity Map (D) theo công thức 
    D = B * f / Z, nơi B là baseline và f là tiêu cự (focal length).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get("enabled", False)
        self.conversion_type = self.config.get("conversion_type", "depth_to_disparity")
        self.baseline = self.config.get("baseline", 0.1) # B (m)
        self.focal_length = self.config.get("focal_length", 700.0) # f (pixels)
        logger.info(f"DisparityConverter initialized. Enabled: {self.enabled}")

    def process(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        if not self.enabled:
            return {"rgb": rgb_image, "depth": depth_map}
        
        output_map = depth_map.copy()
        
        if self.conversion_type == "depth_to_disparity":
            # Chuyển đổi Z -> D: D = B * f / Z
            # Hardening: Tránh chia cho 0
            # Giả định: Depth map đã được chuẩn hóa và các giá trị 0 là invalid/missing
            safe_depth = np.where(output_map > 1e-6, output_map, 1e-6) 
            output_map = (self.baseline * self.focal_length) / safe_depth
            logger.debug("Converted Depth Map to Disparity Map.")
            
        elif self.conversion_type == "disparity_to_depth":
            # Chuyển đổi D -> Z: Z = B * f / D
            safe_disparity = np.where(output_map > 1e-6, output_map, 1e-6)
            output_map = (self.baseline * self.focal_length) / safe_disparity
            logger.debug("Converted Disparity Map to Depth Map.")
        
        else:
            logger.warning(f"Unknown conversion type: {self.conversion_type}. Skipping conversion.")

        return {"rgb": rgb_image, "depth": output_map}