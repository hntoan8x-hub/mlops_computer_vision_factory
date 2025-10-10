import logging
from typing import Dict, Any
from shared_libs.ml_core.retraining.base.base_trigger import BaseTrigger
from shared_libs.monitoring.drift_detector_cv import DriftDetector

logger = logging.getLogger(__name__)

class DriftTrigger(BaseTrigger):
    """
    A trigger that fires when a significant data or prediction drift is detected.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drift_threshold = config.get("drift_threshold", 0.1)
        self.drift_type = config.get("drift_type", "data")
        self.reason_message = ""
        self.drift_detector = DriftDetector(config.get("detector_params"))

    def check(self, **kwargs: Dict[str, Any]) -> bool:
        """
        Checks for drift and returns a boolean trigger signal.

        Args:
            **kwargs: Must contain 'drift_report' from a monitoring service.
        """
        drift_report = kwargs.get("drift_report")
        if not drift_report:
            logger.warning("Drift report not provided. Cannot check for drift.")
            return False

        detected_drift_score = drift_report.get("drift_score", 0.0)
        
        if detected_drift_score > self.drift_threshold:
            self.reason_message = f"Drift detected. Score {detected_drift_score:.4f} exceeds threshold {self.drift_threshold:.4f}."
            logger.warning(self.reason_message)
            return True
            
        return False
        
    def get_reason(self) -> str:
        return self.reason_message