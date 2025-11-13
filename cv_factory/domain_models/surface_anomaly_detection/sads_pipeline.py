# domain_models/surface_anomaly_detection/sads_pipeline.py (HARDENED - Ready for Factory DI)

import logging
from typing import Dict, Any, Union, List, Type
# Import Contracts
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import WorkflowExecutionError
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.monitoring.base_event_emitter import BaseEventEmitter # Giả định Contract

# IMPORT SADS INFERENCE ORCHESTRATOR MỚI
from .sads_inference_orchestrator import SADSInferenceOrchestrator

logger = logging.getLogger(__name__)

class SADSPipeline:
    """
    Façade (mặt tiền) cho Surface Anomaly Detection System (SADS).
    NHẬN các service đã được khởi tạo và TIÊM chúng vào Orchestrator Domain.
    """
    
    def __init__(self, 
                 raw_config: Dict[str, Any], 
                 orchestrator_id: str, 
                 logger_service: BaseTracker,      # <<< INJECTED >>>
                 event_emitter: BaseEventEmitter):  # <<< INJECTED >>>
        """
        Khởi tạo pipeline bằng cách tiêm các service và khởi tạo Orchestrator.
        """
        self.orchestrator_id = orchestrator_id
        self.config = raw_config
        
        # 1. Khởi tạo Orchestrator chuyên biệt (DI các service đã nhận)
        try:
            # GỌI SADSInferenceOrchestrator TRỰC TIẾP
            self.orchestrator: SADSInferenceOrchestrator = SADSInferenceOrchestrator(
                orchestrator_id=orchestrator_id,
                config=raw_config, # Sử dụng config thô (đã được validate bởi BaseOrchestrator)
                logger_service=logger_service,
                event_emitter=event_emitter
            )
            self.logger.info(f"SADS Pipeline initialized via SADSInferenceOrchestrator.")
        except Exception as e:
             self.logger.error(f"Failed to initialize SADSInferenceOrchestrator: {e}")
             raise RuntimeError(f"SADS Pipeline initialization failed: {e}")

    # (Loại bỏ _create_mlops_services_mock và _merge_config_with_domain_details)

    def run_inference(self, raw_input: Any) -> Dict[str, Any]:
        """
        Chạy luồng Sequential Inference.
        """
        if not isinstance(self.orchestrator, SADSInferenceOrchestrator):
            raise RuntimeError("Orchestrator type mismatch. Expected SADSInferenceOrchestrator.")
            
        try:
            # Uỷ quyền toàn bộ việc xử lý cho Orchestrator chuyên biệt
            return self.orchestrator.run(raw_input)
        except WorkflowExecutionError as e:
            self.logger.error(f"SADS Inference failed during execution: {e}")
            raise