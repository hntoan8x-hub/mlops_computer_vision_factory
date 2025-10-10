# cv_factory/shared_libs/ml_core/configs/orchestrator_config_schema.py

from pydantic import Field, validator, BaseModel, NonNegativeInt, constr
from typing import Dict, Any, Union, Literal, Optional, List

# --- Import Schemas from their respective modules ---
# Assuming all schemas below are imported from their final, hardened locations

# Data & Processing Schemas
from shared_libs.data_ingestion.configs.ingestion_config_schema import IngestionConfig
from shared_libs.data_processing.configs.preprocessing_config_schema import ProcessingConfig

# MLOps Logic Schemas
from shared_libs.ml_core.configs.model_config_schema import ModelConfig
from shared_libs.ml_core.configs.trainer_config_schema import TrainerConfig
from shared_libs.ml_core.configs.evaluator_config_schema import EvaluatorConfig
from shared_libs.ml_core.configs.selector_config_schema import SelectorConfig
from shared_libs.ml_core.configs.feature_store_config_schema import FeatureStoreConfig


# --- Base Configuration ---
class BaseConfig(BaseModel):
    class Config:
        # Forbid extra fields at the top level to catch typos early
        extra = "forbid" 
    
    # Run Metadata for Auditing (filled by CI/CD or runtime environment)
    run_metadata: Dict[str, Any] = Field({}, description="Metadata about the execution environment (git_sha, config_hash).")


# --- 1. Training Orchestrator Schema ---

class TrainingOrchestratorConfig(BaseConfig):
    """Master Schema for CVTrainingOrchestrator."""
    
    # 1. Pipeline Definition
    pipeline_type: Literal["training"] = Field("training", description="Must be 'training'.")
    
    # 2. CORE DEPENDENCIES (Mandatory Blocks)
    data_ingestion: IngestionConfig = Field(..., description="Configuration for data I/O and source connections.")
    preprocessing: ProcessingConfig = Field(..., description="Configuration for data transforms, cleaning, and augmentation.")
    model: ModelConfig = Field(..., description="Model architecture and weight configuration.")
    trainer: TrainerConfig = Field(..., description="Trainer execution and DDP setup.")
    evaluator: EvaluatorConfig = Field(..., description="Evaluation metrics and reporting setup.")
    
    # 3. Optional Systems
    feature_store: Optional[FeatureStoreConfig] = Field(None, description="Configuration for feature store integration (if features are saved).")
    
    # CRITICAL: Enforce rule consistency
    @validator('preprocessing')
    def check_augmentation_in_training(cls, v):
        """Rule: Augmentation must be enabled during training for data efficiency."""
        if not v.augmentation.enabled:
            logger.warning("Augmentation is disabled in training configuration. This may lead to overfitting.")
        return v


# --- 2. Inference Orchestrator Schema (Batch & Stream) ---

class InferenceOrchestratorConfig(BaseConfig):
    """Master Schema for CVInferenceOrchestrator and CVStreamInferenceOrchestrator."""
    
    # 1. Pipeline Definition
    pipeline_type: Literal["inference", "stream_inference"] = Field(..., description="Type of inference workflow.")
    
    # 2. CORE DEPENDENCIES (Mandatory Blocks)
    model: ModelConfig = Field(..., description="Model artifact configuration (usually loading from registry).")
    preprocessing: ProcessingConfig = Field(..., description="Preprocessing steps for inference input (MUST be stateless).")
    
    # 3. Optional Systems
    stream_io_config: Optional[Dict[str, Any]] = Field(None, description="Specific config for stream connectors (Kafka/Camera).")
    model_selector: Optional[SelectorConfig] = Field(None, description="Configuration for choosing the best model version.")
    feature_store: Optional[FeatureStoreConfig] = Field(None, description="Configuration for retrieving features during inference.")

    # CRITICAL: Enforce rule consistency
    @validator('preprocessing')
    def check_no_augmentation_in_inference(cls, v):
        """Rule: Augmentation must NOT be enabled during inference."""
        if v.augmentation.enabled:
            raise ValueError("Augmentation must be disabled for inference or stream workflows to ensure deterministic predictions.")
        return v
    
    @validator('stream_io_config')
    def check_stream_config_if_stream_pipeline(cls, v, values):
        """Rule: Stream config is mandatory if pipeline_type is 'stream_inference'."""
        if values.get('pipeline_type') == 'stream_inference' and not v:
            raise ValueError("Pipeline type 'stream_inference' requires 'stream_io_config'.")
        return v


# --- 3. Final Façade (Union Type) ---

# The final class used to validate the input configuration file
MasterOrchestratorConfig = Union[TrainingOrchestratorConfig, InferenceOrchestratorConfig]