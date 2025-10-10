import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor
from shared_libs.ml_core.monitoring.utils.stats_utils import (
    calculate_psi,
    calculate_ks_test
)

logger = logging.getLogger(__name__)

class FeatureDriftMonitor(BaseMonitor):
    """
    Monitors for data drift in the input features.

    It compares the distribution of current features against a reference distribution
    using statistical tests like PSI (Population Stability Index) or KS Test.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drift_threshold = config.get("drift_threshold", 0.1)
        self.check_method = config.get("check_method", "psi")
        self.features_to_monitor = config.get("features_to_monitor", "all")

    def check(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        report = {"drift_detected": False, "feature_reports": {}}
        
        features = self.features_to_monitor
        if features == "all":
            features = list(reference_data.columns)

        for feature in features:
            if feature not in current_data.columns:
                logger.warning(f"Feature '{feature}' not found in current data. Skipping.")
                continue

            ref_feature_data = reference_data[feature]
            curr_feature_data = current_data[feature]

            if self.check_method == "psi":
                score = calculate_psi(ref_feature_data, curr_feature_data)
                report["feature_reports"][feature] = {"score": score, "method": "psi"}
            elif self.check_method == "ks_test":
                score, p_value = calculate_ks_test(ref_feature_data, curr_feature_data)
                report["feature_reports"][feature] = {"score": score, "p_value": p_value, "method": "ks_test"}
            
            if report["feature_reports"][feature]["score"] > self.drift_threshold:
                report["drift_detected"] = True
        
        return report

    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        return report.get("drift_detected", False)

    def get_report_message(self, report: Dict[str, Any]) -> str:
        if report.get("drift_detected", False):
            drifting_features = [f for f, r in report["feature_reports"].items() if r["score"] > self.drift_threshold]
            return f"Data drift detected in the following features: {', '.join(drifting_features)}. Retraining may be needed."
        return "No significant data drift detected."