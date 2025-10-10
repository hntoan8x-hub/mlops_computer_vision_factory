import logging
import numpy as np
from typing import Dict, Any, List

from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor
from shared_libs.ml_core.evaluator.metrics.classification_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

class FairnessMonitor(BaseMonitor):
    """
    Monitors for model bias by comparing performance across different demographic groups.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.group_column = config.get("group_column")
        self.metric_name = config.get("metric_name", "accuracy")
        self.bias_threshold = config.get("bias_threshold", 0.05)
        if self.group_column is None:
            raise ValueError("Group column for fairness check must be specified.")

    def check(self, reference_data: Any, current_data: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        report = {"alert": False, "group_metrics": {}}
        
        y_true_all = kwargs.get("y_true")
        y_pred_all = kwargs.get("y_pred")
        group_labels = current_data[self.group_column]
        
        if y_true_all is None or y_pred_all is None:
            logger.warning("Ground truth or predictions not provided. Cannot check fairness.")
            return report
            
        unique_groups = np.unique(group_labels)
        
        for group in unique_groups:
            group_indices = (group_labels == group)
            y_true_group = y_true_all[group_indices]
            y_pred_group = y_pred_all[group_indices]
            
            group_metrics = compute_classification_metrics(y_true_group, y_pred_group)
            report["group_metrics"][group] = group_metrics
            
        # Check for performance gap between the best and worst performing group
        metric_scores = [metrics.get(self.metric_name, 0.0) for metrics in report["group_metrics"].values()]
        if not metric_scores:
            return report
            
        performance_gap = max(metric_scores) - min(metric_scores)
        if performance_gap > self.bias_threshold:
            report["alert"] = True
            report["performance_gap"] = performance_gap
            
        return report

    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        return report.get("alert", False)

    def get_report_message(self, report: Dict[str, Any]) -> str:
        if report.get("alert", False):
            return f"Fairness alert: Performance gap of {report['performance_gap']:.4f} detected for metric '{self.metric_name}' across groups."
        return "Model appears to be fair across all monitored groups."