# shared_libs/data_labeling/auto_annotation/detection_proposal.py

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
from ...configs.label_schema import DetectionLabel, DetectionObject

logger = logging.getLogger(__name__)

class DetectionProposalAnnotator(BaseAutoAnnotator):
    """
    Annotator chuyên biệt cho Object Detection: Tạo Bounding Boxes và Class IDs.
    """
    
    def _run_inference(self, image_data: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Mô phỏng chạy mô hình YOLO/FasterRCNN và trả về danh sách: 
        [(bbox_xyxy_pixel, class_name, confidence), ...]
        """
        H, W, _ = image_data.shape
        # Giả lập kết quả dự đoán
        predictions = [
            ((int(W*0.1), int(H*0.1), int(W*0.5), int(H*0.5)), "person", 0.95),
            ((int(W*0.6), int(H*0.6), int(W*0.8), int(H*0.8)), "car", 0.80),
            ((int(W*0.2), int(H*0.2), int(W*0.3), int(H*0.3)), "background", 0.50), # Confidence thấp
        ]
        return predictions

    def _normalize_output(self, raw_prediction: List[Tuple[Tuple, str, float]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Chuẩn hóa BBox và áp dụng ngưỡng confidence.
        """
        suggested_objects: List[DetectionObject] = []
        image_path: str = metadata.get("image_path", "unknown")
        
        # Giả định kích thước ảnh có sẵn trong metadata hoặc từ ảnh thô
        img_h, img_w, _ = metadata.get("image_data").shape 

        for bbox_raw, class_name, confidence in raw_prediction:
            if confidence >= self.min_confidence:
                # 1. Chuẩn hóa Bounding Box về [0, 1]
                x_min, y_min, x_max, y_max = bbox_raw
                bbox_normalized: Tuple[float, float, float, float] = (
                    x_min / img_w,
                    y_min / img_h,
                    x_max / img_w,
                    y_max / img_h
                )
                
                # 2. Tạo đối tượng nhãn đã được kiểm tra (Pydantic)
                try:
                    obj = DetectionObject(
                        bbox=bbox_normalized, 
                        class_name=class_name,
                        confidence=confidence # Có thể thêm confidence vào schema DetectionObject
                    )
                    suggested_objects.append(obj)
                except Exception as e:
                    logger.warning(f"Invalid detection object created: {e}")

        # 3. Trả về dưới dạng List[DetectionLabel] (mặc dù chỉ có 1 ảnh)
        return [DetectionLabel(image_path=image_path, objects=suggested_objects)]