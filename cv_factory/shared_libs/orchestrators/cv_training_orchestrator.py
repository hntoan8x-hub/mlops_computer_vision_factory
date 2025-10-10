# cv_factory/shared_libs/orchestrators/cv_training_orchestrator.py (only ml_core/)
# NOTE: The file is located in shared_libs/orchestrators/

import logging
from typing import Dict, Any, Union, Tuple, Optional
import torch
from torch.utils.data import DataLoader, DistributedSampler

# --- Import Contracts and Factories from their final locations ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.ml_core.data.cv_dataset import CVDataset
from shared_libs.ml_core.trainer.factories.trainer_factory import TrainerFactory
from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator
from shared_libs.ml_core.trainer.utils import distributed_utils
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker # Logger Contract
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry # Registry Contract

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
from shared_libs.ml_core.configs.orchestrator_config_schema import TrainingOrchestratorConfig 

# NOTE: We assume the BaseOrchestrator now accepts (id, config, logger_service, event_emitter)

logger = logging.getLogger(__name__)

class CVTrainingOrchestrator(BaseOrchestrator):
    """
    Orchestrates the end-to-end training and evaluation pipeline for a CV model.
    
    This class is the master workflow controller, utilizing injected services 
    and ensuring DDP-ready execution and auditable model registration.
    """
    
    def __init__(self, orchestrator_id: str, config: Dict[str, Any], logger_service: BaseTracker, event_emitter: Any, registry_service: BaseRegistry):
        """
        Initializes the Orchestrator, injecting MLOps services (Logger, Emitter, Registry).
        """
        # 1. Apply Base Orchestrator Init (Handles Structured Logging, Pydantic Validation, Logger/Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        # 2. Inject Registry Service (Needed for registration and tagging)
        self.registry = registry_service 

        # 3. Initialize Factories/Evaluator (Orchestrator's internal tools)
        self.trainer_factory = TrainerFactory
        self.evaluation_orchestrator = EvaluationOrchestrator(
            orchestrator_id=f"{orchestrator_id}-eval",
            config=self.config.get('evaluator', {}) 
        )

    # --- Mandatory Abstract Method Implementation ---

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        [IMPLEMENTED] Validates the configuration using the Pydantic schema.
        
        Raises:
            InvalidConfigError: If the configuration is invalid.
        """
        try:
            # Enforce validation against the strict schema
            TrainingOrchestratorConfig(**config) 
            self.logger.info("Configuration validated successfully against TrainingOrchestratorConfig schema.")
        except Exception as e:
            # CRITICAL: Raise the custom exception for clean error handling
            self.logger.error(f"Invalid training configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Training config validation failed: {e}") from e

    # --- Core Data Flow ---

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepares CVDatasets and creates DDP-compatible DataLoaders.
        """
        self.logger.info(f"[{self.orchestrator_id}] Preparing data: Dataset -> DataLoader.")
        
        # NOTE: Using simplified access to validated config
        train_config = self.config['datasets']['train']
        val_config = self.config['datasets']['validation']
        
        common_dataset_params = {
            'data_connector_config': self.config['data_ingestion']['connector_config'],
            'preprocessing_config': self.config['preprocessing'],
        }

        # Context 'training' enables augmentation; 'evaluation' disables it.
        train_dataset = CVDataset(dataset_id="train_set", config=train_config, context='training', **common_dataset_params)
        val_dataset = CVDataset(dataset_id="validation_set", config=val_config, context='evaluation', **common_dataset_params)

        # 2. Create DataLoaders (DDP-compatible)
        batch_size = self.config['trainer']['batch_size']
        num_workers = self.config['trainer'].get('num_workers', 4)
        is_distributed = distributed_utils.get_world_size() > 1
        
        # Sampler logic must be correct for DDP
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(train_dataset, shuffle=True) if is_distributed else None,
            shuffle=True if not is_distributed else False, # Only shuffle if not DDP
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(val_dataset, shuffle=False) if is_distributed else None,
            shuffle=False,
            num_workers=num_workers
        )
        
        self.logger.info(f"Data preparation complete. Train size: {len(train_dataset)}, DDP Active: {is_distributed}")
        
        return train_loader, val_loader, val_loader # Using val_loader as test_loader

    @measure_orchestrator_latency(orchestrator_type="CVTraining") # Apply Monitoring Decorator
    def run(self) -> Tuple[Dict[str, Any], str]:
        """
        [IMPLEMENTED] Executes the full training workflow: Data -> Train -> Evaluate -> Register Model.
        """
        try:
            # Start a new MLflow run using the injected logger service
            with self.mlops_tracker.start_run(run_name=self.orchestrator_id):
                
                # Log all parameters for reproducibility
                self.log_metrics(self.config) 

                # 1. DATA PREPARATION 
                train_loader, val_loader, test_loader = self._prepare_data()
                
                # 2. TRAINER INSTANTIATION 
                model_config = self.config['model']
                trainer_config = self.config['trainer']
                
                trainer = self.trainer_factory.create(
                    trainer_config['type'], 
                    model_config, 
                    trainer_config
                )

                # 3. TRAINING EXECUTION
                self.logger.info(f"[{self.orchestrator_id}] Starting model training with {trainer_config['type']}...")
                trainer.fit(train_loader, val_loader, epochs=trainer_config['epochs'])
                
                # 4. EVALUATION
                self.logger.info(f"[{self.orchestrator_id}] Starting final evaluation...")
                final_metrics = self.evaluation_orchestrator.run(trainer.model, test_loader)
                
                # Log final metrics
                self.log_metrics({"final_" + k: v for k, v in final_metrics.items()})

                # 5. MODEL REGISTRATION (Only Rank 0 performs saving and registration)
                model_uri = ""
                if distributed_utils.is_main_process():
                    # a. Save model artifact
                    logged_model_uri = self.mlops_tracker.log_model(trainer.model, artifact_path="model")
                    
                    # b. Register the model artifact using the injected registry
                    model_name = model_config.get('name', 'cv_model')
                    model_version = self.registry.register_model(logged_model_uri, model_name)
                    
                    # c. TAGGING (CRITICAL for Auditability)
                    git_sha = self.config.get('run_metadata', {}).get('git_sha', 'unknown')
                    config_hash = self.config.get('run_metadata', {}).get('config_hash', 'unknown')
                    
                    self.registry.tag_model_version(
                        model_name, 
                        model_version, 
                        {'git_sha': git_sha, 'config_hash': config_hash, 'status': 'staging_ready'}
                    )
                    
                    model_uri = f"models:/{model_name}/{model_version}"
                    self.logger.info(f"Model registered and tagged successfully: {model_uri}")
                
                # BARRIER: Ensure all ranks wait for Rank 0 to finish saving/registration
                distributed_utils.synchronize_between_processes("model_registration")

                self.logger.info("End-to-end CV training orchestration completed successfully.")
                return final_metrics, model_uri
            
        except InvalidConfigError:
            # Re-raise InvalidConfigError directly (already logged/emitted)
            raise 
        except Exception as e:
            # CRITICAL: Catch any execution error and wrap it in a custom error
            error_msg = f"Training workflow failed during execution: {type(e).__name__}"
            self.logger.error(error_msg, exc_info=True)
            self.emit_event(event_name="training_workflow_failure", payload={"error": error_msg, "exception": str(e)})
            raise WorkflowExecutionError(error_msg) from e
        finally:
            # Cleanup DDP environment upon exiting the run
            if distributed_utils.get_world_size() > 1:
                distributed_utils.cleanup_distributed_environment()