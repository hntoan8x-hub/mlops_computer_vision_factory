# shared_libs/data_labeling/semi_annotation/methods/active_learning_selector.py

import logging
import random
from typing import List, Dict, Any, Union
import numpy as np
from ..base_semi_annotator import BaseSemiAnnotator
from ....data_labeling.configs.label_schema import StandardLabel

logger = logging.getLogger(__name__)

class ActiveLearningSelector(BaseSemiAnnotator):
    """
    Annotator chuyên biệt cho Active Learning: Lựa chọn mẫu cần được gán nhãn/kiểm tra 
    dựa trên các tiêu chí (ví dụ: độ tin cậy thấp, độ đa dạng).

    Nó giúp tối ưu hóa chi phí gán nhãn bằng cách chỉ chọn những mẫu có giá trị thông tin cao nhất.
    """

    def refine(self, proposals: List[StandardLabel], user_feedback: Union[Dict[str, Any], None] = None) -> List[StandardLabel]:
        """
        [Implement Interface] Không có chức năng tinh chỉnh nhãn. Trả về rỗng.
        """
        return []

    def select_samples(self, pool_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lựa chọn các mẫu dữ liệu cần được gán nhãn/kiểm tra tiếp theo.

        Args:
            pool_metadata (List[Dict]): Toàn bộ metadata của các mẫu chưa được gán nhãn 
                                        hoặc chưa được xác nhận.

        Returns:
            List[Dict]: Danh sách metadata các mẫu được chọn.
        """
        selection_size = self.config.get("selection_size", 100)
        selection_method = self.config.get("method", "random")

        if not pool_metadata:
            return []

        if len(pool_metadata) <= selection_size:
            logger.warning(f"Pool size ({len(pool_metadata)}) is smaller than selection_size. Selecting all samples.")
            return pool_metadata
        
        # --- Logic Lựa Chọn Mẫu ---

        if selection_method == "uncertainty":
            # 1. Uncertainty Sampling (Mô phỏng)
            # Yêu cầu mỗi metadata item phải có trường 'uncertainty_score'
            
            # Giả lập tính toán score nếu chưa có (thực tế cần gọi mô hình)
            if 'uncertainty_score' not in pool_metadata[0]:
                 # Gán score ngẫu nhiên để minh họa
                 for item in pool_metadata:
                     item['uncertainty_score'] = random.random()

            # Chọn những mẫu có score cao nhất (là những mẫu mô hình kém tin cậy nhất)
            sorted_samples = sorted(pool_metadata, key=lambda x: x['uncertainty_score'], reverse=True)
            selected_samples = sorted_samples[:selection_size]
            
            logger.info(f"Active Learning selected {len(selected_samples)} samples using Uncertainty Sampling.")

        elif selection_method == "diversity":
            # 2. Diversity Sampling (Mô phỏng)
            # Logic phức tạp: Yêu cầu vector embedding để tính toán khoảng cách
            
            # Trả về ngẫu nhiên cho mục đích minh họa
            selected_samples = random.sample(pool_metadata, selection_size)
            logger.warning("Diversity Sampling logic is complex and currently defaults to Random Selection.")
            
        elif selection_method == "random":
            # 3. Random Sampling (Mặc định/Fallback)
            selected_samples = random.sample(pool_metadata, selection_size)
            logger.info(f"Active Learning selected {len(selected_samples)} samples using Random Sampling.")
            
        else:
            raise ValueError(f"Unsupported Active Learning method: {selection_method}")

        return selected_samples