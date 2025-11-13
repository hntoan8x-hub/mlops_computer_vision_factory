# shared_libs/ml_core/retraining/retrain_factory.py (NEW FILE)

import logging
from typing import Dict, Any

# Import Contracts & Orchestrator
from shared_libs.ml_core.retraining.orchestrator.retrain_orchestrator import RetrainOrchestrator
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
from shared_libs.infra.monitoring.base_event_emitter import BaseEventEmitter 

logger = logging.getLogger(__name__)

class RetrainOrchestratorFactory:
    """
    Factory chịu trách nhiệm lắp ráp và tiêm các Dependency vào RetrainOrchestrator.
    """

    @staticmethod
    def create(config: Dict[str, Any], 
               logger_service: BaseTracker, 
               registry_service: BaseRegistry, 
               event_emitter: BaseEventEmitter) -> RetrainOrchestrator:
        
        logger.info("Assembling RetrainOrchestrator with MLOps dependencies.")
        
        # NOTE: Orchestrator sẽ tự khởi tạo các Triggers và tiêm registry_service vào chúng.
        orchestrator = RetrainOrchestrator(
            config=config,
            logger_service=logger_service,
            event_emitter=event_emitter,
            registry_service=registry_service
        )
        
        return orchestrator