import logging
from typing import Dict, Any

from shared_libs.ml_core.retraining.base.base_trigger import BaseTrigger

logger = logging.getLogger(__name__)

class PerformanceTrigger(BaseTrigger):
    """
    A trigger that fires when a model's performance metric falls below a threshold.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metric_name = config.get("metric_name")
        self.metric_threshold = config.get("metric_threshold")
        self.mode = config.get("mode", "min") # 'min' for loss, 'max' for accuracy
        self.reason_message = ""

    def check(self, **kwargs: Dict[str, Any]) -> bool:
        current_metrics = kwargs.get("current_metrics")
        if not current_metrics or self.metric_name not in current_metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in current metrics. Cannot check.")
            return False
            
        current_score = current_metrics.get(self.metric_name)
        
        should_trigger = False
        if self.mode == "min" and current_score > self.metric_threshold:
            should_trigger = True
        elif self.mode == "max" and current_score < self.metric_threshold:
            should_trigger = True
            
        if should_trigger:
            self.reason_message = f"Performance drop detected. '{self.metric_name}' score {current_score:.4f} is below threshold {self.metric_threshold:.4f}."
            logger.warning(self.reason_message)
            return True
            
        return False

    def get_reason(self) -> str:
        return self.reason_message