# domain_models/surface_anomaly_detection/sads_postprocessor.py

import logging
import numpy as np
from typing import Dict, Any, List, Union
from .sads_config_schema import SADSPostprocessorParams # Import tham số nghiệp vụ

logger = logging.getLogger(__name__)

# NOTE: Giả định utility NMS có sẵn
def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Mô phỏng hàm Non-Maximum Suppression."""
    # (Implementation chi tiết của NMS cần được đặt trong shared_libs/core_utils/numpy_utils)
    # Trả về chỉ mục của các hộp không bị trùng lặp
    return np.arange(len(boxes)) # Trả về tất cả cho mục đích mô phỏng

class SADSPostprocessor:
    """
    Lớp Postprocessor Domain-specific cho Surface Anomaly Detection.
    Được Inject vào CVPredictor để áp dụng Business Logic (NMS, PASS/FAIL Decision).
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Khởi tạo với tham số nghiệp vụ (được tải qua Dynamic Import).
        """
        try:
            # Pydantic validation của tham số nghiệp vụ
            self.params = SADSPostprocessorParams(**kwargs)
        except Exception as e:
            raise ValueError(f"SADS Postprocessor Config validation failed: {e}")

        logger.info(f"SADS Postprocessor initialized. Confidence Threshold: {self.params.defect_confidence_threshold}")

    def _make_decision(self, defect_count: int, defect_area_ratio: float) -> str:
        """
        Decision Engine: Quyết định PASS/FAIL dựa trên Business Rules.
        """
        # Rule 1: Số lượng lỗi vượt quá giới hạn
        if self.params.max_allowed_defects == 0 and defect_count > 0:
            return "FAIL"
        if defect_count > self.params.max_allowed_defects and self.params.max_allowed_defects > 0:
            return "FAIL"
        
        # Rule 2: Diện tích lỗi vượt quá giới hạn
        if defect_area_ratio > self.params.min_defect_area_normalized:
            return "FAIL"
            
        return "PASS"

    def run(self, 
            standardized_predictions: List[Dict[str, Any]], 
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Thực thi Business Logic trên đầu ra đã được chuẩn hóa từ OutputAdapter.

        Args:
            standardized_predictions (List[Dict]): Output từ OutputAdapter: 
                                                   List of [{'box': array, 'score': float, 'class': int}]
            config (Dict): Cấu hình bổ sung từ Predictor (không cần thiết ở đây).

        Returns:
            Dict[str, Any]: Kết quả cuối cùng (có Decision, BBoxes đã NMS).
        """
        
        if not standardized_predictions:
            return {"decision": "PASS", "defects": [], "summary": {"count": 0, "area_ratio": 0.0}}

        boxes = np.array([p['box'] for p in standardized_predictions])
        scores = np.array([p['score'] for p in standardized_predictions])
        classes = np.array([p['class'] for p in standardized_predictions])

        # 1. Lọc ngưỡng Confidence
        valid_indices = scores >= self.params.defect_confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        classes = classes[valid_indices]
        
        # 2. Áp dụng NMS (Non-Maximum Suppression)
        if len(boxes) > 0:
            nms_indices = apply_nms(boxes, scores, self.params.nms_iou_threshold)
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]
            classes = classes[nms_indices]
        
        # 3. Tổng hợp kết quả và tính toán diện tích
        final_defects: List[Dict[str, Any]] = []
        total_defect_area_normalized = 0.0

        for box, score, cls in zip(boxes, scores, classes):
            # Tính diện tích (giả định BBox là normalized [0, 1])
            area = (box[2] - box[0]) * (box[3] - box[1]) 
            total_defect_area_normalized += area
            
            final_defects.append({
                "type": f"defect_{int(cls)}",
                "score": float(score),
                "bbox_normalized": box.tolist()
            })

        # 4. Quyết định (Decision)
        decision = self._make_decision(len(final_defects), total_defect_area_normalized)

        return {
            "decision": decision,
            "defects": final_defects,
            "summary": {
                "count": len(final_defects),
                "area_ratio": total_defect_area_normalized
            }
        }