# shared_libs/ml_core/monitoring/reporters/alert_reporter.py

import logging
from typing import Dict, Any
from shared_libs.ml_core.monitoring.base.base_reporter import BaseReporter
from shared_libs.ml_core.monitoring.utils.alert_utils import send_slack_alert, send_email_alert # Import utilities

logger = logging.getLogger(__name__)

class AlertReporter(BaseReporter):
    """
    A concrete reporter that sends critical alerts via external channels (Slack, Email).
    
    It uses configuration defined in monitoring_config_schema.AlertConfig.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slack_webhook_url = self.config.get("slack_webhook_url")
        self.email_recipients = self.config.get("email_recipients", [])
        
        if not self.slack_webhook_url and not self.email_recipients:
            logger.warning("AlertReporter initialized without Slack or Email configured. Alerts will be logged only.")
        else:
             logger.info("AlertReporter initialized and configured.")

    def report(self, report_name: str, report_data: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        """
        Reports an outcome by sending notifications if an alert is detected.
        
        Args:
            report_name (str): The name of the report (e.g., "fairness_alert").
            report_data (Dict[str, Any]): The full report data (including 'message').
        """
        alert_message = report_data.get("message", f"Monitor {report_name} triggered an alert.")
        
        if not report_data.get("alert", False):
            return # Only send notifications if an alert is explicitly flagged
        
        full_message = f"🚨 MLOps ALERT: {report_name.upper()} 🚨\nDetails: {alert_message}\nData: {report_data}"
        
        if self.slack_webhook_url:
            send_slack_alert(full_message, self.slack_webhook_url)
            
        if self.email_recipients:
            send_email_alert(f"[CRITICAL] MLOps Alert: {report_name}", full_message, self.email_recipients)
            
        logger.warning(f"CRITICAL ALERT handled: {report_name}")