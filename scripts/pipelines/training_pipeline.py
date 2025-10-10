# cv_factory/domain_models/medical_imaging/pipelines/training_pipeline.py

import logging
from typing import Dict, Any

# Import Platform components (ML Core)
from shared_libs.ml_core.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.ml_core.mlflow_service.implementations.mlflow_registry import MLflowRegistry
from shared_libs.ml_core.configs.config_utils import load_config # Assuming a config utility

# Import Domain-Specific Components
from ..configs.training_config import TrainingConfigSchema # Assuming Pydantic schema for validation
# NOTE: The actual training_config.yaml is the input here

logger = logging.getLogger(__name__)

class MedicalTrainingPipeline:
    """
    A domain-specific adapter for the Medical Imaging training workflow.

    This class loads the domain configuration and invokes the generic 
    CVTrainingOrchestrator from the shared platform library. It DOES NOT 
    contain any training loop logic.
    """

    def __init__(self, run_id: str, config_path: str = "medical_imaging/configs/training_config.yaml"):
        """
        Initializes the pipeline by loading and validating the domain configuration.

        Args:
            run_id (str): A unique identifier for the current training run.
            config_path (str): Path to the domain-specific training configuration file.
        """
        self.run_id = run_id
        self.config_path = config_path
        self.full_config: Dict[str, Any] = {}
        self.registry = MLflowRegistry() # Initialize platform component
        
        # 1. Load and Validate Domain Configuration
        try:
            raw_config = load_config(self.config_path) # Load YAML
            
            # Use Pydantic schema to validate the structure (Best practice!)
            # This relies on the Pydantic check we added in the CI/CD step
            # TrainingConfigSchema(**raw_config) 
            
            self.full_config = raw_config
            logger.info(f"[{self.run_id}] Medical training config loaded and validated.")
        except Exception as e:
            logger.error(f"Failed to load or validate training configuration: {e}")
            raise

    def run(self):
        """
        Executes the training workflow by delegating the task to the platform orchestrator.
        """
        if not self.full_config:
            raise RuntimeError("Configuration is empty. Cannot run pipeline.")

        logger.info(f"[{self.run_id}] Starting medical image training pipeline...")

        # 1. INITIALIZE PLATFORM ORCHESTRATOR
        # The orchestrator is responsible for creating the DataLoaders, Trainers, and Loggers.
        training_orchestrator = CVTrainingOrchestrator(
            orchestrator_id=f"medical-train-{self.run_id}",
            config=self.full_config
        )

        # 2. EXECUTE TRAINING
        try:
            # The orchestrator handles the entire MLOps lifecycle:
            # Data Ingestion -> Preprocessing -> Training (DDP) -> Evaluation -> Model Registration
            final_metrics, model_uri = training_orchestrator.run_training_flow()
            
            logger.info(f"[{self.run_id}] Pipeline completed. Final model URI: {model_uri}")
            logger.info(f"Final Metrics: {final_metrics}")
            
            return final_metrics, model_uri
            
        except Exception as e:
            logger.error(f"Training Orchestrator failed during execution: {e}")
            # Optionally, log failure to MLflow/Notification Service
            raise