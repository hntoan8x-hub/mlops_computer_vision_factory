import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

def send_slack_alert(message: str, webhook_url: str) -> None:
    """
    Sends an alert message to a Slack channel via webhook.
    """
    try:
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info("Slack alert sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")