import logging
import numpy as np
from typing import Dict, Any

from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor
from shared_libs.ml_core.monitoring.utils.stats_utils import calculate_jensen_shannon_divergence

logger = logging.getLogger(__name__)

class SHAPMonitor(BaseMonitor):
    """
    Monitors for drift in feature importance by comparing SHAP value distributions.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drift_threshold = config.get("drift_threshold", 0.1)
        
    def check(self, reference_data: np.ndarray, current_data: np.ndarray, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        report = {"alert": False}
        
        # Reference and current data are assumed to be SHAP value distributions
        # For simplicity, we'll use a statistical test to compare them.
        kl_divergence = calculate_jensen_shannon_divergence(reference_data, current_data)
        
        if kl_divergence > self.drift_threshold:
            report["alert"] = True
            report["kl_divergence"] = kl_divergence
        
        return report

    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        return report.get("alert", False)

    def get_report_message(self, report: Dict[str, Any]) -> str:
        if report.get("alert", False):
            return f"SHAP value drift detected. KL Divergence is {report['kl_divergence']:.4f}, exceeding the threshold {self.drift_threshold:.4f}."
        return "No significant SHAP value drift detected."