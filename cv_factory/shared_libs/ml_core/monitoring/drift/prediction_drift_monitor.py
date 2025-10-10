import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor
from shared_libs.ml_core.monitoring.utils.stats_utils import calculate_kl_divergence

logger = logging.getLogger(__name__)

class PredictionDriftMonitor(BaseMonitor):
    """
    Monitors for drift in the model's prediction distribution.
    
    It compares the distribution of current predictions against a reference distribution
    using a statistical measure like KL Divergence.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drift_threshold = config.get("drift_threshold", 0.05)
        self.check_method = config.get("check_method", "kl_divergence")

    def check(self, reference_data: np.ndarray, current_data: np.ndarray, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        report = {"drift_detected": False}
        
        # Ensure data is in the form of distributions (histograms or probabilities)
        # This is a conceptual example
        reference_hist, _ = np.histogram(reference_data, bins=10, density=True)
        current_hist, _ = np.histogram(current_data, bins=10, density=True)

        if self.check_method == "kl_divergence":
            score = calculate_kl_divergence(reference_hist, current_hist)
            report["score"] = score
        
        if report["score"] > self.drift_threshold:
            report["drift_detected"] = True
            
        return report

    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        return report.get("drift_detected", False)

    def get_report_message(self, report: Dict[str, Any]) -> str:
        if report.get("drift_detected", False):
            return f"Prediction drift detected. KL Divergence score {report['score']:.4f} exceeds threshold {self.drift_threshold:.4f}."
        return "No significant prediction drift detected."