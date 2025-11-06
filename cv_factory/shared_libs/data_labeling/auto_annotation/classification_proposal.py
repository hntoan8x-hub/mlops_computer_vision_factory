# shared_libs/data_labeling/auto_annotation/classification_proposal.py

import logging
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from torch import Tensor

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
from ...configs.label_schema import ClassificationLabel

logger = logging.getLogger(__name__)

class ClassificationProposalAnnotator(BaseAutoAnnotator):
    """
    Annotator chuyên biệt cho Image Classification: Gán nhãn lớp cho toàn bộ ảnh.
    Sử dụng mô hình như CLIP, ViT hoặc EfficientNet để dự đoán.
    """
    
    def _run_inference(self, image_data: np.ndarray) -> Tuple[str, float]:
        """
        Mô phỏng chạy mô hình phân loại và trả về lớp dự đoán + độ tin cậy.

        Args:
            image_data (np.ndarray): Ảnh đầu vào (H, W, C).

        Returns:
            Tuple[str, float]: (Tên lớp dự đoán, Độ tin cậy)
        """
        # Giả lập kết quả dự đoán (thực tế cần gọi self.model.predict)
        
        # Ví dụ: Giả lập mô hình dự đoán lớp "dog"
        if np.mean(image_data) > 128:
            predicted_class = "cat"
            confidence = 0.92
        else:
            predicted_class = "dog"
            confidence = 0.85
            
        return predicted_class, confidence

    def _normalize_output(self, raw_prediction: Tuple[str, float], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Chuẩn hóa kết quả dự đoán thô thành đối tượng ClassificationLabel (Pydantic).
        """
        predicted_class, confidence = raw_prediction
        image_path: str = metadata.get("image_path", "unknown")
        
        # 1. Áp dụng ngưỡng confidence
        if confidence < self.min_confidence:
            logger.warning(f"Skipping auto-label for {image_path}: confidence ({confidence:.2f}) is below threshold ({self.min_confidence:.2f}).")
            return [] # Trả về rỗng nếu độ tin cậy thấp

        # 2. Tạo đối tượng nhãn đã được kiểm tra (Pydantic)
        try:
            # ClassificationLabel chỉ yêu cầu image_path và label (tên lớp)
            # Confidence là thông tin bổ sung có thể lưu trong metadata hoặc cấu trúc label chi tiết hơn.
            label_obj = ClassificationLabel(
                image_path=image_path,
                label=predicted_class
            )
            
            # Chúng ta có thể bổ sung thông tin confidence vào cấu trúc Dict trả về
            # nếu cần theo dõi chất lượng nhãn.
            label_dict = label_obj.model_dump()
            label_dict['confidence'] = confidence
            
            # NOTE: Phải trả về một List[StandardLabel] để nhất quán với BaseAnnotator
            return [label_dict]
            
        except Exception as e:
            logger.error(f"Failed to create valid ClassificationLabel for {image_path}: {e}")
            return []