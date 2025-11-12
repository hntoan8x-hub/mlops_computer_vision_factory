# shared_libs/data_processing/mask_components/mask_validator.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_mask_processor import BaseMaskProcessor, MaskData

logger = logging.getLogger(__name__)

class MaskValidator(BaseMaskProcessor):
    """
    Component chịu trách nhiệm xác thực (Validate) Mask/Label Map sau khi xử lý 
    (ví dụ: kiểm tra class distribution, kiểm tra mask bị rỗng).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_class_ratio = self.config.get("min_class_ratio", 0.001) # Tối thiểu 0.1% pixel phải có lớp
        self.required_classes = self.config.get("required_classes", [])
        logger.info(f"MaskValidator initialized. Min class ratio: {self.min_class_ratio}")

    def process(self, rgb_image: np.ndarray, mask_data: MaskData) -> Dict[str, Any]:
        
        current_mask = mask_data
        total_pixels = 0
        
        if isinstance(current_mask, np.ndarray) and current_mask.ndim < 3:
            # Xử lý Segmentation/Label Map (Long Label)
            total_pixels = current_mask.size
            
            # 1. Kiểm tra Empty Mask (Bỏ qua class 0/background)
            unique_classes = np.unique(current_mask[current_mask != 0])
            if len(unique_classes) == 0:
                logger.warning("Mask is empty (contains only background class 0).")
                
            # 2. Kiểm tra phân phối class tối thiểu
            for c in unique_classes:
                pixel_count = np.sum(current_mask == c)
                ratio = pixel_count / total_pixels
                if ratio < self.min_class_ratio:
                    logger.warning(
                        f"Class {c} has low pixel coverage: {ratio:.4f} < {self.min_class_ratio}."
                    )
            
        elif isinstance(current_mask, dict) and 'boxes' in current_mask:
            # Xử lý Bounding Boxes
            num_boxes = len(current_mask['boxes'])
            if num_boxes == 0:
                logger.warning("BBox list is empty (no objects detected).")
                
            # 3. Kiểm tra các lớp bắt buộc (nếu có)
            if self.required_classes:
                labels = set(current_mask['labels'])
                missing_labels = set(self.required_classes) - labels
                if missing_labels:
                    logger.error(f"Missing required labels in sample: {missing_labels}")
                    
        logger.debug("Mask data passed validation.")
        
        return {"rgb": rgb_image, "mask": current_mask}