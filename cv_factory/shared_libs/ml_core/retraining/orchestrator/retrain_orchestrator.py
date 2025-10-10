# shared_libs/ml_core/retraining/orchestrator/retrain_orchestrator.py

import logging
import importlib
from typing import Dict, Any, List, Type, Optional

# --- Import Contracts and Utilities ---
from shared_libs.ml_core.retraining.base.base_retrain_orchestrator import BaseRetrainOrchestrator
from shared_libs.ml_core.retraining.base.base_trigger import BaseTrigger
from shared_libs.ml_core.retraining.configs.retrain_config_schema import RetrainConfig
from shared_libs.ml_core.retraining.utils.job_utils import submit_training_job
from shared_libs.ml_core.retraining.utils.notification_utils import send_slack_alert
from shared_libs.core_utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class RetrainOrchestrator(BaseRetrainOrchestrator):
    """
    The orchestrator that manages the retraining lifecycle:
    1. Checks all configured triggers.
    2. Submits a new training job if any trigger fires.
    3. Handles notifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            self.validated_config = RetrainConfig(**config)
        except Exception as e:
            raise ConfigurationError(f"Retraining configuration validation failed: {e}") from e
            
        self.triggers: List[BaseTrigger] = self._instantiate_triggers()
        self.notification_config = self.validated_config.notification
        
        logger.info(f"Retrain Orchestrator initialized. Active Triggers: {len(self.triggers)}")

    @staticmethod
    def _get_trigger_class(trigger_type: str) -> Type[BaseTrigger]:
        """Dynamically loads the specific Trigger class based on type name."""
        # Convention: 'drift' -> 'DriftTrigger', 'time' -> 'TimeTrigger', etc.
        class_name = "".join(word.capitalize() for word in trigger_type.split("_")) + "Trigger"
        module_path = f"shared_libs.ml_core.retraining.triggers.{trigger_type}_trigger"
        
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            raise ImportError(f"Cannot load trigger class '{class_name}' from {module_path}. Error: {e}")

    def _instantiate_triggers(self) -> List[BaseTrigger]:
        """Instantiates all configured triggers dynamically."""
        triggers = []
        
        for trigger_config in self.validated_config.triggers:
            try:
                TriggerCls = self._get_trigger_class(trigger_config.type)
                triggers.append(TriggerCls(trigger_config.params))
            except Exception as e:
                logger.error(f"Failed to instantiate trigger '{trigger_config.type}': {e}")
                continue
        return triggers

    def run(self, **kwargs: Dict[str, Any]) -> None:
        """
        Executes the retraining decision logic (Check Triggers) and submits the job if needed.
        """
        trigger_fired = False
        reasons = []
        
        # 1. CHECK ALL TRIGGERS
        for trigger in self.triggers:
            if trigger.check(**kwargs):
                trigger_fired = True
                reasons.append(trigger.get_reason())
                
        # 2. DECISION LOGIC
        if trigger_fired:
            logger.warning(f"RETRAINING TRIGGERED! Reasons: {', '.join(reasons)}")
            
            # 3. SUBMIT JOB
            job_config = self.validated_config.job_config
            submit_training_job(
                training_config_path=job_config.training_config_path,
                reasons=reasons,
                **job_config.params
            )
            
            # 4. NOTIFICATION
            message = f"🚨 RETRAINING JOB SUBMITTED 🚨\nReasons: {', '.join(reasons)}"
            send_slack_alert(message, self.notification_config.slack_webhook_url)
            
            self.log_job_status("submitted", reasons=reasons)
        else:
            logger.info("No retraining triggers fired. System status OK.")
            self.log_job_status("check_completed_no_trigger")

    def log_job_status(self, status: str, **kwargs: Dict[str, Any]) -> None:
        """
        [IMPLEMENTED] Logs the status of the retraining job to a logging/monitoring service.
        """
        # In a real system, this would call an injected MLOps Logger (like MLflowLogger)
        log_payload = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "orchestrator_id": self.__class__.__name__,
            **kwargs
        }
        logger.info(f"RETRAIN STATUS: {log_payload}")