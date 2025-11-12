# cv_factory/shared_libs/ml_core/evaluator/factories/metric_factory.py

import logging
from typing import Dict, Any, Type, Optional
from ..base.base_metric import BaseMetric

# Import Metrics cho Classification
from ..metrics.classification_metrics import AccuracyMetric, F1ScoreMetric
# Import Metrics cho Detection
from ..metrics.detection_metrics import mAPMetric
# Import Metrics cho Segmentation
from ..metrics.segmentation_metrics import DiceCoefficientMetric, MeanIoUMetric
# Import Metrics cho OCR
from ..metrics.ocr_metrics import CharacterErrorRateMetric 
# Import Metrics cho Embedding
from ..metrics.embedding_metrics import RecallAtKMetric
# --- IMPORT MỚI CHO DEPTH ESTIMATION ---
from ..metrics.depth_metrics import AbsRelMetric, ThresholdAccuracyMetric

logger = logging.getLogger(__name__)

class MetricFactory:
    """
    Factory class quản lý việc tạo ra các Metric có trạng thái (Stateful Metrics) 
    dựa trên tên metric.
    """
    
    # Ánh xạ tên metric (string) sang Metric Class
    METRIC_MAPPING: Dict[str, Type[BaseMetric]] = {
        # Classification
        "accuracy": AccuracyMetric,
        "f1_score": F1ScoreMetric,
        
        # Detection
        "map": mAPMetric,
        
        # Segmentation
        "dice_coefficient": DiceCoefficientMetric,
        "mean_iou": MeanIoUMetric,
        
        # OCR
        "cer": CharacterErrorRateMetric,
        
        # Embedding/Retrieval
        "recall@k": RecallAtKMetric,
        
        # --- MỚI: Depth Estimation ---
        "abs_rel": AbsRelMetric, # Lưu trữ AbsRel, SqRel, RMSE
        "delta_accuracy": ThresholdAccuracyMetric, # Lưu trữ Delta_1, Delta_2, Delta_3
        # Lưu ý: Các tên metric khác (SqRel, RMSE) sẽ được truy cập qua khóa của AbsRelMetric.compute()
    }

    @staticmethod
    def get_metric(metric_name: str, config: Optional[Dict[str, Any]] = None) -> BaseMetric:
        """
        Tạo và trả về một Metric instance.

        Args:
            metric_name (str): Tên của metric cần khởi tạo (ví dụ: 'mAP', 'accuracy').
            config (Optional[Dict[str, Any]]): Cấu hình cho metric (ví dụ: 'iou_thresholds', 'K').

        Returns:
            BaseMetric: Một instance của Metric class tương ứng.
        """
        metric_name = metric_name.lower()
        if metric_name not in MetricFactory.METRIC_MAPPING:
            raise ValueError(f"Unsupported metric name: '{metric_name}'. "
                             f"Available metrics: {list(MetricFactory.METRIC_MAPPING.keys())}")

        MetricClass = MetricFactory.METRIC_MAPPING[metric_name]
        
        try:
            return MetricClass(name=metric_name, config=config)
        except Exception as e:
            logger.error(f"Failed to instantiate Metric '{metric_name}': {e}")
            raise RuntimeError(f"Metric creation failed: {e}")