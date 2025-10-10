from pydantic import Field, NonNegativeInt, validator, constr
from typing import Dict, Any, Optional
from shared_libs.ml_core.configs.base_config_schema import BaseConfig
from shared_libs.data_ingestion.configs.ingestion_config_schema import IngestionConfig
from shared_libs.data_processing.configs.preprocessing_config_schema import ProcessingConfig
# NOTE: Need to import/define Trainer, Model, and Evaluator schemas similarly

# Placeholder Schemas for brevity, assuming they inherit BaseConfig
class ModelConfig(BaseConfig):
    name: str = Field(..., description="Model name (e.g., 'resnet50').")
    pretrained: bool = Field(False, description="Load pre-trained weights.")

class TrainerConfig(BaseConfig):
    type: constr(to_lower=True) = Field(..., description="Trainer type (e.g., 'cnn_trainer', 'transformer_trainer').")
    batch_size: NonNegativeInt = Field(32, description="Training batch size.")
    epochs: NonNegativeInt = Field(10, description="Number of training epochs.")
    
class EvaluatorConfig(BaseConfig):
    metrics: List[str] = Field(..., description="List of metric names to compute (e.g., 'accuracy', 'map').")

# --- MASTER ORCHESTRATOR SCHEMAS ---

class TrainingOrchestratorConfig(BaseConfig):
    """Master Schema for CVTrainingOrchestrator."""
    pipeline_type: constr(to_lower=True) = Field("training", const=True, description="Must be 'training'.")
    
    # Dependencies
    data_ingestion: IngestionConfig = Field(..., description="Configuration for data I/O.")
    preprocessing: ProcessingConfig = Field(..., description="Configuration for data transforms and features.")
    model: ModelConfig = Field(..., description="Model architecture configuration.")
    trainer: TrainerConfig = Field(..., description="Trainer execution configuration.")
    evaluator: EvaluatorConfig = Field(..., description="Evaluation setup.")

class InferenceOrchestratorConfig(BaseConfig):
    """Master Schema for CVInferenceOrchestrator and CVStreamInferenceOrchestrator."""
    pipeline_type: constr(to_lower=True) = Field(..., description="Pipeline type ('inference' or 'stream_inference').")
    
    # Key dependencies for inference
    model: ModelConfig = Field(..., description="Model configuration (used to determine model artifact).")
    preprocessing: ProcessingConfig = Field(..., description="Preprocessing for inference input.")
    
    # Stream-specific config
    stream_io_config: Optional[Dict[str, Any]] = Field(None, description="Specific config for stream connectors (Kafka/Camera).")


# Final Façade to validate against any top-level config
class OrchestratorConfig(BaseConfig):
    """
    Facade schema to validate the entire configuration file against the appropriate pipeline type.
    """
    
    @validator('pipeline_type', pre=True, always=True)
    def determine_schema(cls, v, values):
        # Dynamically switch validation based on the pipeline type field
        pipeline_type = values.get('pipeline', {}).get('type', 'training').lower()
        
        if pipeline_type == 'training':
            return TrainingOrchestratorConfig(**values)
        elif pipeline_type in ['inference', 'stream_inference']:
            return InferenceOrchestratorConfig(**values)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return v