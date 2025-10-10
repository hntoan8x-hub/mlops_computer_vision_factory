# cv_factory/shared_libs/orchestrators/cv_stream_inference_orchestrator.py
# NOTE: File is located in shared_libs/orchestrators/

import logging
import time
from typing import Dict, Any, Optional, Iterator, Union

# --- Import Abstractions and Contracts from their final locations ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.inference.base_cv_predictor import BaseCVPredictor, PredictionOutput 
from shared_libs.data_ingestion.base.base_stream_connector import BaseStreamConnector
from shared_libs.data_ingestion.factories.stream_connector_factory import StreamConnectorFactory
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
from shared_libs.ml_core.configs.orchestrator_config_schema import InferenceOrchestratorConfig 

logger = logging.getLogger(__name__)

class CVStreamInferenceOrchestrator(BaseOrchestrator):
    """
    Orchestrates the continuous real-time inference pipeline on streaming data (e.g., Kafka, Camera).

    This class manages the lifecycle (connect/close) of the Stream Connector, enforces 
    validation, and delegates prediction to the injected Predictor.
    """

    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 predictor: BaseCVPredictor,
                 logger_service: BaseTracker,
                 event_emitter: Any):
        """
        Initializes the Stream Inference Orchestrator with all required MLOps dependencies.
        
        Args:
            predictor (BaseCVPredictor): The model predictor instance (INJECTED).
            logger_service (BaseTracker): The service for logging metrics/params (INJECTED).
            event_emitter (Any): The service for emitting structured events (INJECTED).
        """
        # 1. Apply Base Orchestrator Init (Handles Validation, Logger/Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.predictor = predictor # Store the injected dependency
        self.stream_config = self.config.get('stream_io_config', {})
        self.input_connector: Optional[BaseStreamConnector] = None
        
        # Determine output topics/paths from config
        self.output_destination = self.stream_config.get('output_destination')
        self.connector_type = self.stream_config.get('connector_type', 'kafka')
        
        # CRITICAL: Validate that the injected Predictor is loaded before running the loop
        if not self.predictor.is_loaded:
            raise RuntimeError("Predictor model must be loaded before running the stream loop.")
            
        self._initialize_connector()
        self.logger.info(f"[{self.orchestrator_id}] CV Stream Orchestrator initialized and ready.")

    # --- Mandatory Abstract Method Implementation (Validation) ---
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        [IMPLEMENTED] Validates the configuration using the Pydantic schema 
        (Enforces Quality Gate).
        
        Raises:
            InvalidConfigError: If the configuration is invalid.
        """
        try:
            # Enforce validation against the strict schema
            InferenceOrchestratorConfig(**config) 
            self.logger.info("Configuration validated successfully against Stream Inference Config schema.")
        except Exception as e:
            # CRITICAL: Raise the custom exception for clean error handling
            self.logger.error(f"Invalid stream configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Stream config validation failed: {e}") from e

    def _initialize_connector(self):
        """
        Initializes the appropriate Stream Connector using the Factory.
        """
        try:
            connector_config = self.stream_config.get('connector_config', {})
            
            self.input_connector = StreamConnectorFactory.get_stream_connector(
                connector_type=self.connector_type,
                connector_config=connector_config,
                connector_id=f"{self.orchestrator_id}-stream-connector"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Stream Connector: {e}")
            self.emit_event(event_name="stream_connector_setup_failure", payload={"error": str(e)})
            raise RuntimeError(f"Orchestrator setup failed: {e}")

    @measure_orchestrator_latency(orchestrator_type="CVStreamInference") # Apply Monitoring Decorator
    def run(self, max_runtime_seconds: Optional[int] = None) -> None:
        """
        [IMPLEMENTED] Executes the main real-time inference loop.
        """
        if not self.input_connector:
            raise RuntimeError("Input Stream Connector is not available.")

        self.logger.info(f"Starting real-time inference loop on connector: {self.connector_type}...")
        start_time = time.time()
        frame_count = 0
        
        try:
            # Use Context Manager to ensure safe connection/disconnection (CRITICAL for Kafka/Camera)
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
                        # CRITICAL: Log event and continue loop (Stream processing should be fault-tolerant)
                        error_msg = f"Error processing frame/message {frame_count}."
                        self.logger.error(f"[{self.orchestrator_id}] {error_msg}. Skipping.", exc_info=True)
                        self.emit_event(event_name="stream_processing_error", payload={"error": str(e), "frame": frame_count})
                        continue # Skip the current frame/message and continue the loop

        except Exception as e:
            # Catch exceptions that break the entire loop (e.g., Kafka connection lost, Camera handle failed)
            error_msg = f"Fatal stream workflow failure: {type(e).__name__}."
            self.logger.error(f"[{self.orchestrator_id}] {error_msg}", exc_info=True)
            self.emit_event(event_name="stream_workflow_failure_fatal", payload={"error": error_msg, "exception": str(e)})
            raise WorkflowExecutionError(error_msg) from e # Wrap and raise for structured handling
        
        self.logger.info(f"Stream inference loop finished. Total items processed: {frame_count}")