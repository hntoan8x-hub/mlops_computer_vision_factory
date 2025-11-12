# shared_libs/data_processing/depth_components/depth_augmenter.py
import logging
import numpy as np
import random
from typing import Dict, Any, Optional
from .base_depth_processor import BaseDepthProcessor

logger = logging.getLogger(__name__)

class DepthAugmenter(BaseDepthProcessor):
    """
    Component áp dụng các kỹ thuật Augmentation đồng bộ cho cả RGB và Depth Map 
    (ví dụ: lật, xoay, thêm nhiễu).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.apply_flip = self.config.get("apply_flip", False)
        self.add_noise = self.config.get("add_noise", False)
        
        # HARDENING: Config cho deterministic seed
        self.deterministic_seed = self.config.get("seed", None)
        logger.info(f"DepthAugmenter initialized. Flip: {self.apply_flip}, Noise: {self.add_noise}")

    def process(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        
        # HARDENING: Ensure deterministic seed in DDP training
        if self.deterministic_seed is not None:
            np.random.seed(self.deterministic_seed)
            random.seed(self.deterministic_seed)
        
        current_rgb = rgb_image
        current_depth = depth_map
        
        # 1. Đồng bộ Horizontal Flip
        if self.apply_flip and random.random() > 0.5:
            current_rgb = np.flip(current_rgb, axis=1).copy()
            current_depth = np.flip(current_depth, axis=1).copy()
            logger.debug("Applied Horizontal Flip.")

        # 2. Thêm nhiễu chỉ vào Depth (vì nó nhạy cảm hơn)
        if self.add_noise:
            # Ví dụ: Gaussian Noise nhẹ
            noise_std = self.config.get("noise_std", 0.01)
            noise = np.random.normal(0, noise_std, current_depth.shape).astype(current_depth.dtype)
            current_depth = current_depth + noise
            logger.debug(f"Added Gaussian Noise (std={noise_std}) to Depth Map.")
        
        return {"rgb": current_rgb, "depth": current_depth}