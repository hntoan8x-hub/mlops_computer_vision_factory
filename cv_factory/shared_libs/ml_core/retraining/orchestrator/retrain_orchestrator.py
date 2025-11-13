# shared_libs/ml_core/retraining/orchestrator/retrain_orchestrator.py (UPDATED)

import logging
import importlib
from typing import Dict, Any, List, Type, Optional
from datetime import datetime
from shared_libs.ml_core.retraining.base.base_retrain_orchestrator import BaseRetrainOrchestrator
from shared_libs.ml_core.retraining.base.base_trigger import BaseTrigger
from shared_libs.ml_core.retraining.configs.retrain_config_schema import RetrainConfig
from shared_libs.ml_core.retraining.utils.job_utils import submit_training_job
from shared_libs.ml_core.retraining.utils.notification_utils import send_slack_alert
from shared_libs.core_utils.exceptions import ConfigurationError

# NEW IMPORTS for DI
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
from shared_libs.infra.monitoring.base_event_emitter import BaseEventEmitter 
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry 

logger = logging.getLogger(__name__)

class RetrainOrchestrator(BaseRetrainOrchestrator):
    """
    The orchestrator that manages the retraining lifecycle:
    1. Checks all configured triggers.
    2. Submits a new training job if any trigger fires.
    3. Handles notifications.
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 logger_service: BaseTracker, # <<< INJECTED >>>
                 event_emitter: BaseEventEmitter, # <<< INJECTED >>>
                 registry_service: BaseRegistry # <<< INJECTED for Triggers >>>
                ):
        
        super().__init__(config)
        
        try:
            self.validated_config = RetrainConfig(**config)
        except Exception as e:
            raise ConfigurationError(f"Retraining configuration validation failed: {e}") from e
            
        self.mlops_tracker = logger_service # Store injected tracker
        self.emitter = event_emitter
        self.registry = registry_service
        
        # CẬP NHẬT: Truyền registry_service vào khi khởi tạo triggers
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
        """Instantiates all configured triggers dynamically, injecting the Registry."""
        triggers = []
        
        for trigger_config in self.validated_config.triggers:
            try:
                TriggerCls = self._get_trigger_class(trigger_config.type)
                
                # CẬP NHẬT: Tiêm registry_service vào Trigger
                triggers.append(
                    TriggerCls(
                        trigger_config.params, 
                        registry_service=self.registry # <<< DI REGISTRY >>>
                    )
                )
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
            # Truyền thêm các tham số cần thiết cho check() (ví dụ: drift_report, current_metrics)
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
        [HARDENED] Logs the status of the retraining job to the injected MLOps Tracker.
        """
        log_payload = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "orchestrator_id": self.__class__.__name__,
            **kwargs
        }
        
        # NEW: Log job status vào MLOps Tracker (MLflow)
        # Sử dụng log_params để đảm bảo các metadata được lưu trữ dưới dạng key-value
        self.mlops_tracker.log_params(log_payload) 
        
        # NEW: Emit event (Ví dụ: để kích hoạt một quy trình khác)
        self.emitter.emit_event(f"retrain_job_{status}", log_payload)
        
        logger.info(f"RETRAIN STATUS: {log_payload}")