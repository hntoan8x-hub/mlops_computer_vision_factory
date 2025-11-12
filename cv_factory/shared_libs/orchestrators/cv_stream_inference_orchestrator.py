# cv_factory/shared_libs/orchestrators/cv_stream_inference_orchestrator.py (HARDENED)

import logging
import time
from typing import Dict, Any, Optional, Iterator, Union

# --- Import Abstractions and Contracts from their final locations ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.inference.base_cv_predictor import BaseCVPredictor, PredictionOutput 
from shared_libs.data_ingestion.base.base_stream_connector import BaseStreamConnector
# LOẠI BỎ: from shared_libs.data_ingestion.factories.stream_connector_factory import StreamConnectorFactory
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
from shared_libs.ml_core.configs.orchestrator_config_schema import InferenceOrchestratorConfig 

logger = logging.getLogger(__name__)

class CVStreamInferenceOrchestrator(BaseOrchestrator):
    """
    Orchestrates the continuous real-time inference pipeline on streaming data (e.g., Kafka, Camera).

    HARDENED: Accepts an injected BaseCVPredictor and a pre-initialized BaseStreamConnector.
    """

    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 predictor: BaseCVPredictor,
                 input_connector: BaseStreamConnector, # <<< INJECTED CONNECTOR >>>
                 logger_service: BaseTracker,
                 event_emitter: Any):
        """
        Initializes the Stream Inference Orchestrator with all required MLOps dependencies.
        """
        
        # 1. Apply Base Orchestrator Init (Handles Validation, Logger/Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.predictor = predictor # Store the injected dependency
        self.input_connector: BaseStreamConnector = input_connector # Store the INJECTED connector
        
        # Determine output topics/paths from config (stream_config can still be useful)
        self.stream_config = self.config.get('stream_io_config', {})
        self.output_destination = self.stream_config.get('output_destination')
        self.connector_type = type(input_connector).__name__
        
        # CRITICAL: Validate that the injected Predictor is loaded
        if not self.predictor.is_loaded:
            raise RuntimeError("Predictor model must be loaded by the Factory before running the stream loop.")
            
        # LOẠI BỎ: self._initialize_connector()
        self.logger.info(f"[{self.orchestrator_id}] CV Stream Orchestrator initialized with connector {self.connector_type} and ready.")

    # --- Mandatory Abstract Method Implementation (Validation) ---
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        [IMPLEMENTED] Validates the configuration using the Pydantic schema.
        """
        try:
            InferenceOrchestratorConfig(**config) 
            self.logger.info("Configuration validated successfully against Stream Inference Config schema.")
        except Exception as e:
            self.logger.error(f"Invalid stream configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Stream config validation failed: {e}") from e

    # LOẠI BỎ: def _initialize_connector(self):

    @measure_orchestrator_latency(orchestrator_type="CVStreamInference") # Apply Monitoring Decorator
    def run(self, max_runtime_seconds: Optional[int] = None) -> None:
        """
        [IMPLEMENTED] Executes the main real-time inference loop using the injected connector.
        """
        # KHÔNG cần kiểm tra self.input_connector vì nó đã được tiêm qua __init__
        # if not self.input_connector:
        #     raise RuntimeError("Input Stream Connector is not available.")

        self.logger.info(f"Starting real-time inference loop on injected connector: {self.connector_type}...")
        start_time = time.time()
        frame_count = 0
        
        try:
            # Use Context Manager on the INJECTED connector
            with self.input_connector as connector:
                for raw_data in connector.consume():
                    current_time = time.time()
                    
                    if max_runtime_seconds is not None and (current_time - start_time) > max_runtime_seconds:
                        self.logger.info(f"Max runtime ({max_runtime_seconds}s) reached. Stopping stream loop.")
                        break

                    try:
                        # 1. PREDICT: Run the full inference pipeline
                        prediction_result: PredictionOutput = self.predictor.predict_pipeline(raw_data)

                        # 2. PRODUCE: Write the result back to the output stream/destination
                        if self.output_destination:
                            connector.produce(
                                data=prediction_result, 
                                destination_topic=self.output_destination,
                                key=str(frame_count)
                            )
                            self.log_metrics({"processed_frame_count": frame_count + 1}) 
                            
                        frame_count += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing frame/message {frame_count}."
                        self.logger.error(f"[{self.orchestrator_id}] {error_msg}. Skipping.", exc_info=True)
                        self.emit_event(event_name="stream_processing_error", payload={"error": str(e), "frame": frame_count})
                        continue 

        except Exception as e:
            error_msg = f"Fatal stream workflow failure: {type(e).__name__}."
            self.logger.error(f"[{self.orchestrator_id}] {error_msg}", exc_info=True)
            self.emit_event(event_name="stream_workflow_failure_fatal", payload={"error": error_msg, "exception": str(e)})
            raise WorkflowExecutionError(error_msg) from e 
        
        self.logger.info(f"Stream inference loop finished. Total items processed: {frame_count}")