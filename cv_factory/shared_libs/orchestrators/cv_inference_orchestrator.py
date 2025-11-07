# cv_factory/shared_libs/orchestrators/cv_inference_orchestrator.py


import logging
from typing import Dict, Any, Union, List
import numpy as np

# --- Import Contracts and Factories from their final locations ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.inference.base_cv_predictor import BaseCVPredictor, PredictionOutput 
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
from shared_libs.ml_core.configs.orchestrator_config_schema import InferenceOrchestratorConfig 

# NOTE: We assume the BaseOrchestrator handles initialization of its logger instance (self.logger)

class CVInferenceOrchestrator(BaseOrchestrator):
    """
    Orchestrates the model inference pipeline for batch or single-request CV models.
    
    This class acts as a thin facade, utilizing the injected BaseCVPredictor, 
    enforcing Pydantic configuration validation, and integrating Prometheus monitoring.
    """
    
    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 predictor: BaseCVPredictor,
                 logger_service: BaseTracker,
                 event_emitter: Any):
        """
        Initializes the Inference Orchestrator with all required MLOps dependencies.
        """
        # 1. Apply Base Orchestrator Init (Handles Validation, Logger/Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.config = config
        self.predictor = predictor # Store the injected dependency

        # 2. Enforce Model Loading (Lifecycle Management)
        if not self.predictor.is_loaded:
            self.logger.info("Predictor model not yet loaded. Attempting to load from config...")
            model_uri = self.config.get('model', {}).get('uri')
            if model_uri:
                self.predictor.load_model(model_uri)
            else:
                # CRITICAL: Throw runtime error if no model source is configured
                raise RuntimeError("Predictor must be loaded or configured with a model URI before running.")
        
        self.logger.info(f"[{self.orchestrator_id}] CV Inference Orchestrator initialized and model is ready.")

    # --- Mandatory Abstract Method Implementation ---
    
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
            self.logger.info("Configuration validated successfully against InferenceOrchestratorConfig schema.")
        except Exception as e:
            # CRITICAL: Raise the custom exception for clean error handling
            self.logger.error(f"Invalid inference configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Inference config validation failed: {e}") from e

    @measure_orchestrator_latency(orchestrator_type="CVInference") # Apply Monitoring Decorator
    def run(self, inputs: Union[np.ndarray, List[np.ndarray], List[Dict[str, Any]]]) -> List[PredictionOutput]:
        """
        [IMPLEMENTED] Runs the full inference workflow on the given input data (Batch or Single Request).
        """
        self.logger.info(f"[{self.orchestrator_id}] Starting batch CV inference orchestration.")
        
        if not inputs:
            self.logger.warning(f"[{self.orchestrator_id}] Received empty input list.")
            return []
            
        final_predictions: List[PredictionOutput] = []

        try:
            # 1. PREDICT: Delegate the entire pipeline to the injected predictor
            prediction_result = self.predictor.predict_pipeline(inputs)
            
            # 2. LOGGING: Log the completion and number of predictions
            self.log_metrics({"inference_count": len(inputs)})
            self.logger.info(f"[{self.orchestrator_id}] Inference orchestration completed for {len(inputs)} items.")
            
            # 3. Finalize output list
            if isinstance(prediction_result, list):
                final_predictions.extend(prediction_result)
            else:
                final_predictions.append(prediction_result)

        except Exception as e:
            # CRITICAL: Wrap exception in WorkflowExecutionError for upstream handling
            error_msg = f"Inference pipeline failed: {type(e).__name__}"
            self.logger.error(f"[{self.orchestrator_id}] {error_msg}", exc_info=True)
            self.emit_event(event_name="inference_failure", payload={"error": error_msg, "orchestrator_id": self.orchestrator_id})
            raise WorkflowExecutionError(error_msg) from e

        return final_predictions