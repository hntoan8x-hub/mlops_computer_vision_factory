# cv_factory/shared_libs/ml_core/evaluator/factories/metric_factory.py

import logging
from typing import Dict, Any, Type, Union

# Import Base Abstraction
from ..base.base_metric import BaseMetric

# Import Concrete Classification Metrics
from ..metrics.classification_metrics import AccuracyMetric, F1ScoreMetric

# Import Concrete Detection Metrics
from ..metrics.detection_metrics import mAPMetric

# Import Concrete Segmentation Metrics (assuming future implementation)
# from ..metrics.segmentation_metrics import IoUMetric, DiceCoefficientMetric 

logger = logging.getLogger(__name__)

class MetricFactory:
    """
    Factory class responsible for creating concrete instances of BaseMetric.
    
    This centralizes the instantiation of all stateful evaluation metrics used 
    throughout the CV pipeline (training, evaluation, and monitoring).
    """
    
    # Mapping of metric names to their concrete class implementations
    METRIC_MAPPING: Dict[str, Type[BaseMetric]] = {
        # Classification
        "accuracy": AccuracyMetric,
        "f1_score": F1ScoreMetric,
        
        # Detection
        "map": mAPMetric,
        
        # Segmentation (Placeholders)
        # "iou_seg": IoUMetric,
        # "dice": DiceCoefficientMetric,
    }

    @staticmethod
    def get_metric(metric_name: str, config: Union[Dict[str, Any], None] = None) -> BaseMetric:
        """
        Creates and returns a concrete stateful metric instance.

        Args:
            metric_name (str): The name of the metric to create (e.g., 'accuracy', 'map').
            config (Dict[str, Any], optional): Configuration dictionary specific to the metric 
                                               (e.g., IoU thresholds for mAP, average type for F1).

        Returns:
            BaseMetric: An instance of the requested concrete metric class.

        Raises:
            ValueError: If the metric_name is not supported.
            RuntimeError: If instantiation fails.
        """
        metric_name = metric_name.lower()
        config = config if config is not None else {}
        
        if metric_name not in MetricFactory.METRIC_MAPPING:
            raise ValueError(
                f"Unsupported metric type: '{metric_name}'. "
                f"Available metrics are: {list(MetricFactory.METRIC_MAPPING.keys())}"
            )
            
        MetricClass = MetricFactory.METRIC_MAPPING[metric_name]
        
        try:
            # Instantiate the metric, passing name and configuration
            metric = MetricClass(
                name=metric_name,
                config=config
            )
            logger.info(f"Successfully created stateful metric: {metric_name}")
            return metric
        except Exception as e:
            logger.error(f"Failed to instantiate metric '{metric_name}': {e}")
            raise RuntimeError(f"Metric creation failed for '{metric_name}': {e}")