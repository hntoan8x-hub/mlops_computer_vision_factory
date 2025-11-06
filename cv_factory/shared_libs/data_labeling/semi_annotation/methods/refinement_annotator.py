# shared_libs/data_labeling/semi_annotation/methods/refinement_annotator.py

import logging
from typing import List, Dict, Any, Union
from ..base_semi_annotator import BaseSemiAnnotator
from ....data_labeling.configs.label_schema import StandardLabel

logger = logging.getLogger(__name__)

class RefinementAnnotator(BaseSemiAnnotator):
    """
    Annotator chuyên biệt cho việc Tinh chỉnh nhãn (Refinement).
    Sử dụng logic hoặc feedback để cải thiện nhãn đề xuất.
    """

    def refine(self, 
               proposals: List[StandardLabel], 
               user_feedback: Union[Dict[str, Any], None] = None
    ) -> List[StandardLabel]:
        """
        Logic: Hợp nhất (NMS), loại bỏ overlap, hoặc áp dụng sửa đổi từ người dùng.
        """
        final_labels: List[StandardLabel] = []
        
        if user_feedback and user_feedback.get("action") == "accepted":
            # Nếu người dùng chấp nhận, sử dụng nhãn đề xuất hoặc nhãn đã sửa đổi
            final_labels = proposals # Hoặc user_feedback.get("corrected_labels")
            logger.info("Labels accepted and finalized by user feedback.")
        elif user_feedback and user_feedback.get("action") == "rejected":
            # Nếu người dùng từ chối, bỏ qua đề xuất
            logger.warning("Labels rejected by user feedback.")
            final_labels = []
        else:
            # Logic làm sạch tự động (ví dụ: Non-Maximum Suppression (NMS) cho BBox)
            # Giả định: chỉ giữ lại các nhãn có độ tin cậy cao nhất hoặc không trùng lặp
            final_labels = [p for p in proposals if p.model_dump().get("confidence", 1.0) >= self.config.get("final_threshold", 0.9)]
            logger.info(f"Applied automated refinement: filtered {len(proposals) - len(final_labels)} proposals.")

        return final_labels

    def select_samples(self, pool_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Không có chức năng chọn mẫu. Trả về rỗng."""
        return []

