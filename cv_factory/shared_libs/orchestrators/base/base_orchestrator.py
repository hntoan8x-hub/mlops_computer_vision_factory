import abc
import logging
from typing import Dict, Any, Optional
import json

# --- IMPORT UTILITIES MỚI ---
from shared_libs.orchestrators.utils.orchestrator_logging import get_orchestrator_logger 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
# Import Type Hints for the expected Services (Logging and Eventing)
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
from shared_libs.infra.monitoring.base_event_emitter import BaseEventEmitter # Giả định BaseEventEmitter tồn tại

# NOTE: Logger is initialized with the custom utility
# We will initialize the specific instance inside __init__
# logger = logging.getLogger(__name__) 

class BaseOrchestrator(abc.ABC):
    """
    Abstract Base Class for all ML workflow orchestrators.

    Enforces integration with MLOps tracking and eventing services via Dependency Injection,
    and utilizes Structured Logging and Custom Exceptions.
    """

    def __init__(self, orchestrator_id: str, config: Dict[str, Any], logger_service: BaseTracker, event_emitter: Any):
        """
        Initializes the Orchestrator with configuration and required MLOps services.
        
        Args:
            orchestrator_id (str): A unique identifier for the orchestration run.
            config (Dict[str, Any]): The configuration dictionary.
            logger_service (BaseTracker): The service responsible for logging metrics/params.
            event_emitter (Any): The service responsible for emitting structured events.
        """
        # 1. Initialize Custom Structured Logger
        self.logger = get_orchestrator_logger(orchestrator_id)
        
        # 2. Inject MLOps Services
        self.mlops_tracker = logger_service  # Use a different name to avoid conflict with structured logger
        self.emitter = event_emitter
        
        self.orchestrator_id = orchestrator_id
        self.config = config
        
        # Enforce validation immediately upon initialization
        self.validate_config(config)
        self.logger.info(f"{self.__class__.__name__} initialized with validated configuration and services.")

    @abc.abstractmethod
    def run(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Executes the main workflow of the orchestrator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        [MANDATORY] Validates the structure and content of the configuration dictionary 
        using a Pydantic schema appropriate for the concrete subclass.
        
        Raises:
            InvalidConfigError: If the configuration is invalid.
        """
        # Subclasses should raise InvalidConfigError upon Pydantic failure
        raise NotImplementedError 

    def log_metrics(self, metrics: Dict[str, Any], **kwargs: Dict[str, Any]) -> None:
        """
        Logs a set of metrics to the injected MLOps tracking service (e.g., MLflow).
        """
        if self.mlops_tracker:
            self.mlops_tracker.log_metrics(metrics, **kwargs)
        else:
            self.logger.warning("MLOps Tracker not initialized. Metrics not logged.")
        
    def emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Emits a structured event to the injected event emission service and logs it locally.
        
        Args:
            event_name (str): The name of the event (e.g., 'workflow_start', 'inference_failure').
            payload (Dict[str, Any]): The event payload.
        """
        # 1. Log event locally using Structured Logger (for audit trail)
        log_data = {"event_name": event_name, "payload": payload}
        self.logger.info(json.dumps(log_data))

        # 2. Emit event to external system (Prometheus/Kafka/etc.)
        if self.emitter:
            self.emitter.emit_event(event_name, payload)