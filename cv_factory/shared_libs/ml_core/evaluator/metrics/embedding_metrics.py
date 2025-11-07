# cv_factory/shared_libs/ml_core/evaluator/metrics/embedding_metrics.py

import numpy as np
import logging
from typing import Dict, Any, Union, List, Optional

from ..base.base_metric import BaseMetric, MetricValue, InputData
from sklearn.metrics import pairwise_distances # Utility cho khoảng cách

logger = logging.getLogger(__name__)

class RecallAtKMetric(BaseMetric):
    """
    Metric Recall@K: Tỷ lệ các truy vấn (query) tìm thấy ít nhất một positive match 
    trong K kết quả gần nhất. Phổ biến trong Face Recognition, Image Retrieval.
    """
    
    def __init__(self, name: str = 'Recall@K', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.K = self.config.get('K', 1) # Mặc định là Recall@1
        self.distance_metric = self.config.get('distance_metric', 'euclidean')
        self.reset()

    def reset(self) -> None:
        """
        Đặt lại trạng thái tích lũy các vector embedding và ID/nhãn tương ứng.
        """
        self._internal_state = {
            'embeddings': [],  # List[np.ndarray]
            'labels': [],      # List[int/str]
        }
        self._is_initialized = True

    def update(self, embeddings: InputData, labels: InputData, **kwargs) -> None:
        """
        Tích lũy các vector embedding và nhãn ID/Class.
        
        Args:
            embeddings (np.ndarray): Các vector đặc trưng.
            labels (np.ndarray): Các ID thực thể/nhãn tương ứng.
        """
        if not self._is_initialized: self.reset()
        
        # Đảm bảo input là numpy arrays
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # Xử lý nếu input là batch (ví dụ: embeddings.shape = [B, D])
        if embeddings.ndim == 2:
            self._internal_state['embeddings'].extend(list(embeddings))
            self._internal_state['labels'].extend(list(labels))
        elif embeddings.ndim == 1:
            self._internal_state['embeddings'].append(embeddings)
            self._internal_state['labels'].append(labels.item() if labels.size == 1 else labels[0])
        else:
            raise ValueError("Embeddings must be 1D or 2D array.")

    def compute(self) -> float:
        """
        Tính toán Recall@K bằng cách so sánh từng embedding với tất cả embedding khác.
        """
        if len(self._internal_state['embeddings']) < 2:
            return 0.0

        embeddings_array = np.array(self._internal_state['embeddings'])
        labels_array = np.array(self._internal_state['labels'])
        
        # 1. Tính ma trận khoảng cách (Distance Matrix)
        # Ma trận D[i, j] là khoảng cách giữa embedding i và embedding j
        distance_matrix = pairwise_distances(embeddings_array, metric=self.distance_metric)
        
        correct_retrievals = 0
        total_queries = len(embeddings_array)
        
        for i in range(total_queries):
            # 2. Tìm K kết quả gần nhất (trừ chính nó)
            # Sắp xếp các chỉ số theo khoảng cách tăng dần (nhỏ nhất là gần nhất)
            # [1:] để loại bỏ chính nó (khoảng cách = 0)
            sorted_indices = np.argsort(distance_matrix[i])[1:self.K + 1] 
            
            # 3. Kiểm tra xem có Positive Match nào trong K kết quả không
            query_label = labels_array[i]
            
            # Kiểm tra nếu bất kỳ nhãn nào trong K kết quả gần nhất khớp với nhãn truy vấn
            is_match_found = any(labels_array[j] == query_label for j in sorted_indices)
            
            if is_match_found:
                correct_retrievals += 1

        # 4. Tính Recall@K
        recall_at_k = correct_retrievals / total_queries
        
        # Trả về Dict để nhất quán với các metrics phức tạp khác
        return {
            self.name: float(recall_at_k),
            "K_value": self.K
        }