import logging
import numpy as np
from typing import Dict, Any

from shared_libs.ml_core.monitoring.base.base_monitor import BaseMonitor

logger = logging.getLogger(__name__)

class LatencyMonitor(BaseMonitor):
    """
    Monitors the inference latency of a deployed model.
    
    It checks if the average latency exceeds a specified threshold.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.latency_threshold_ms = config.get("latency_threshold_ms", 100)
        
    def check(self, reference_data: Any, current_data: Dict[str, Any], **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        report = {"alert": False}
        current_latency = current_data.get("average_latency_ms")
        
        if current_latency is None:
            logger.warning("Average latency not found in current data.")
            report["alert"] = False
            return report
        
        if current_latency > self.latency_threshold_ms:
            report["alert"] = True
            report["latency"] = current_latency
            report["threshold"] = self.latency_threshold_ms
            
        return report

    def get_alert_status(self, report: Dict[str, Any]) -> bool:
        return report.get("alert", False)

    def get_report_message(self, report: Dict[str, Any]) -> str:
        if report.get("alert", False):
            return f"High latency detected. Average latency is {report['latency']:.2f}ms, exceeding the threshold of {report['threshold']}ms."
        return "Inference latency is within the acceptable range."