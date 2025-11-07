# cv_factory/shared_libs/ml_core/evaluator/metrics/ocr_metrics.py

import numpy as np
import logging
from typing import Dict, Any, Union, List, Optional
from jiwer import wer, cer # Giả định sử dụng thư viện jiwer hoặc tương đương

from ..base.base_metric import BaseMetric, MetricValue, InputData

logger = logging.getLogger(__name__)

class CharacterErrorRateMetric(BaseMetric):
    """
    Metric tính Tỷ lệ Lỗi Ký tự (CER). 
    CER = (Substitutions + Insertions + Deletions) / Total Characters
    """
    
    def __init__(self, name: str = 'CER', config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.reset()

    def reset(self) -> None:
        """Đặt lại trạng thái tích lũy các cặp (Ground Truth, Prediction)."""
        self._internal_state = {
            'all_truths': [],  # List[str]
            'all_predictions': [] # List[str]
        }
        self._is_initialized = True

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Tích lũy các cặp chuỗi dự đoán và chuỗi ground truth.

        Args:
            predictions (List[str]): Chuỗi văn bản dự đoán.
            targets (List[str]): Chuỗi văn bản ground truth.
        """
        if not self._is_initialized: self.reset()
            
        if not isinstance(predictions, list) or not isinstance(targets, list):
             raise TypeError("OCR metrics require input as List[str].")
             
        self._internal_state['all_predictions'].extend(predictions)
        self._internal_state['all_truths'].extend(targets)

    def compute(self) -> float:
        """
        Tính toán CER cuối cùng trên toàn bộ tập dữ liệu.
        """
        if not self._internal_state['all_truths']:
            return 0.0
            
        try:
            # Tính CER (dùng hàm từ thư viện bên ngoài)
            # Hàm cer(truth, prediction)
            score = cer(self._internal_state['all_truths'], self._internal_state['all_predictions'])
            return float(score)
        except Exception as e:
            logger.error(f"Error computing CER: {e}")
            return 1.0 # Trả về 1.0 (100% lỗi) nếu có lỗi tính toán