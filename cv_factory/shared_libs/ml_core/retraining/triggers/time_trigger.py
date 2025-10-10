import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from shared_libs.ml_core.retraining.base.base_trigger import BaseTrigger

logger = logging.getLogger(__name__)

class TimeTrigger(BaseTrigger):
    """
    A trigger that fires based on a fixed time interval.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.last_retrain_time = datetime.fromisoformat(config.get("last_retrain_time", "1970-01-01T00:00:00"))
        self.interval_days = config.get("interval_days", 7)
        self.reason_message = ""

    def check(self, **kwargs: Dict[str, Any]) -> bool:
        current_time = datetime.now()
        next_retrain_time = self.last_retrain_time + timedelta(days=self.interval_days)
        
        if current_time >= next_retrain_time:
            self.reason_message = f"Time-based trigger fired. Next retrain scheduled for {next_retrain_time.isoformat()}."
            logger.info(self.reason_message)
            return True
            
        return False

    def get_reason(self) -> str:
        return self.reason_message