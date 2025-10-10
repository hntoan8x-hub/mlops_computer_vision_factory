import logging
import numpy as np
from typing import Dict, Any

from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor

logger = logging.getLogger(__name__)

class MetricMonitor(BaseMonitor):
    """
    Monitors a model's performance metrics against a defined threshold.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metric_name = config.get("metric_name")
        self.threshold = config.get("threshold")
        self.mode = config.get("mode", "min")  # 'min' for loss, 'max' for accuracy
        if self.metric_name is None or self.threshold is None:
            raise ValueError("Metric name and threshold must be specified.")
        
    def check(self, reference_data: Any, current_data: Dict[str, Any], **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        report = {"alert": False}
        current_score = current_data.get(self.metric_name)
        
        if current_score is None:
            logger.warning(f"Metric '{self.metric_name}' not found in current data.")
            report["alert"] = False
            return report

        if (self.mode == "max" and current_score < self.threshold) or \
           (self.mode == "min" and current_score > self.threshold):
            report["alert"] = True
            report["score"] = current_score
            report["threshold"] = self.threshold
        
        return report

    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        return report.get("alert", False)

    def get_report_message(self, report: Dict[str, Any]) -> str:
        if report.get("alert", False):
            return f"Performance drop detected. Metric '{self.metric_name}' score is {report['score']:.4f}, below the threshold {report['threshold']:.4f}."
        return f"Model performance for '{self.metric_name}' is within the acceptable range."