# shared_libs/orchestrators/cv_training_orchestrator.py (FINAL UPDATE WITH LIVE DATA INJECTION)

import logging
from typing import Dict, Any, Union, Tuple, Optional, Type
import torch
import os 
from torch.utils.data import DataLoader # Cần cho type hint

# --- Import Contracts and Factories ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.ml_core.dataset.cv_dataset import CVDataset
from shared_libs.ml_core.trainer.trainer_factory import TrainerFactory 
from shared_libs.ml_core.trainer.base.base_trainer import BaseTrainer 
from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator
from shared_libs.ml_core.trainer.utils import distributed_utils
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry 
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory 
from shared_libs.orchestrators.cv_deployment_orchestrator import CVDeploymentOrchestrator 
from shared_libs.ml_core.dataloader.dataloader_factory import DataLoaderFactory 

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
# Note: Giả định TrainingOrchestratorConfig được import/định nghĩa tại đây
from shared_libs.ml_core.configs.orchestrator_config_schema import TrainingOrchestratorConfig 

logger = logging.getLogger(__name__)

class CVTrainingOrchestrator(BaseOrchestrator):
    """
    Orchestrates the end-to-end training and evaluation pipeline for a CV model.
    HARDENED: Uses Dependency Injection for all complex components (Evaluator, Deployer, Factories).
    """
    
    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 logger_service: BaseTracker, 
                 event_emitter: Any, 
                 registry_service: BaseRegistry,
                 
                 evaluation_orchestrator: EvaluationOrchestrator,
                 deployment_orchestrator: Optional[CVDeploymentOrchestrator],
                 trainer_factory: Type[TrainerFactory],
                 ml_component_factory: Type[ComponentFactory],
                 live_data_uri_override: Optional[str] = None # <<< THAM SỐ GHI ĐÈ URI MỚI >>>
                ):
        
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.registry = registry_service 
        self.evaluation_orchestrator = evaluation_orchestrator 
        self.deployment_orchestrator = deployment_orchestrator 
        self.trainer_factory = trainer_factory                 
        self.ml_component_factory = ml_component_factory       
        self.live_data_uri_override = live_data_uri_override # LƯU TRỮ URI GHI ĐÈ
        
        self.logger.info("CVTrainingOrchestrator initialized with all required services INJECTED.")


    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validates the configuration using the Pydantic schema TrainingOrchestratorConfig.
        """
        try:
            TrainingOrchestratorConfig(**config) 
            self.logger.info("Configuration validated successfully against TrainingOrchestratorConfig schema.")
        except Exception as e:
            self.logger.error(f"Invalid training configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Training config validation failed: {e}") from e


    def _prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepares CVDatasets and creates DDP-compatible DataLoaders.
        UPDATED: Pass live_data_uri_override to CVDataset.
        """
        self.logger.info(f"[{self.orchestrator_id}] Preparing data: Dataset -> DataLoader.")
        
        train_config = self.config['datasets']['train']
        val_config = self.config['datasets']['validation']
        
        common_dataset_params = {
            'data_connector_config': self.config['data_ingestion']['connector_config'],
            'preprocessing_config': self.config['preprocessing'],
            'labeling_config': self.config['labeling'],
            'ml_component_factory': self.ml_component_factory,
            'live_data_uri_override': self.live_data_uri_override # <<< TRUYỀN URI GHI ĐÈ XUỐNG DATASET >>>
        }

        # 1. Khởi tạo Datasets
        # Dataset sẽ xử lý việc ghi đè URI khi khởi tạo Data Connector nội bộ.
        train_dataset = CVDataset(dataset_id="train_set", config=train_config, context='training', **common_dataset_params)
        val_dataset = CVDataset(dataset_id="validation_set", config=val_config, context='evaluation', **common_dataset_params)

        # 2. DELEGATION: Ủy quyền việc tạo DataLoader cho Factory
        train_loader = DataLoaderFactory.create(
            train_dataset, self.config, context='training'
        )
        val_loader = DataLoaderFactory.create(
            val_dataset, self.config, context='evaluation'
        )
        
        self.logger.info(f"Data preparation complete. Train size: {len(train_dataset)}, DataLoader Factory used.")
        
        return train_loader, val_loader, val_loader # Dùng val_loader làm test_loader

    @measure_orchestrator_latency(orchestrator_type="CVTraining")
    def run(self) -> Tuple[Dict[str, Any], str]:
        """
        Executes the full training workflow: Data -> Train -> Evaluate -> Register -> DEPLOY.
        """
        logged_model_uri = ""
        endpoint_id = "" 
        
        try:
            # BẮT ĐẦU MLFLOW RUN
            with self.mlops_tracker.start_run(run_name=self.orchestrator_id):
                
                self.mlops_tracker.log_params(self.config) 

                # 1. DATA PREPARATION 
                train_loader, val_loader, test_loader = self._prepare_data()
                
                # 2. TRAINER INSTANTIATION (Delegation to INJECTED TrainerFactory)
                model_config = self.config['model']
                trainer_config = self.config['trainer']
                trainer = self.trainer_factory.create(
                    trainer_config['type'], 
                    model_config, 
                    trainer_config
                )

                # 3. TRAINING EXECUTION
                self.logger.info(f"[{self.orchestrator_id}] Starting model training...")
                trainer.fit(train_loader, val_loader, epochs=trainer_config['epochs'])
                
                # 4. EVALUATION
                self.logger.info(f"[{self.orchestrator_id}] Starting final evaluation...")
                final_metrics = self.evaluation_orchestrator.evaluate(trainer.model, test_loader)
                self.mlops_tracker.log_metrics({f"final_{k}": v for k, v in final_metrics['metrics'].items()})
                
                # 5. MODEL REGISTRATION 
                if distributed_utils.is_main_process():
                    logged_model_uri = self.mlops_tracker.log_model(trainer.model, artifact_path="model")
                    
                    model_name = self.config['model'].get('name', 'cv_model')
                    model_version = self.registry.register_model(logged_model_uri, model_name)
                    
                    git_sha = self.config.get('run_metadata', {}).get('git_sha', 'unknown')
                    config_hash = self.config.get('run_metadata', {}).get('config_hash', 'unknown')
                    self.registry.tag_model_version(
                        model_name, 
                        model_version, 
                        {'git_sha': git_sha, 'config_hash': config_hash, 'status': 'staging_ready'}
                    )
                    
                    logged_model_uri = f"models:/{model_name}/{model_version}"
                    self.logger.info(f"Model registered and tagged successfully: {logged_model_uri}")

                logged_model_uri = distributed_utils.broadcast_and_wait(logged_model_uri)
                
                # 6. TRIỂN KHAI MÔ HÌNH 
                if distributed_utils.is_main_process() and self.deployment_orchestrator:
                    self.logger.info(f"[{self.orchestrator_id}] Kích hoạt Deployment Orchestrator (INJECTED)...")
                    
                    endpoint_id = self.deployment_orchestrator.run(
                        model_artifact_uri=logged_model_uri,
                        model_name=model_name,
                        mode=self.config.get('deployment', {}).get('initial_mode', 'standard'),
                        new_version_tag=model_version,
                        stable_version=self.config.get('deployment', {}).get('stable_version', 'latest') 
                    )
                    self.logger.info(f"Deployment to endpoint '{endpoint_id}' completed.")

                self.logger.info("End-to-end CV training orchestration completed successfully.")
                return final_metrics, endpoint_id
            
        except InvalidConfigError:
            raise 
        except Exception as e:
            error_msg = f"Training workflow failed during execution: {type(e).__name__}"
            self.logger.error(error_msg, exc_info=True)
            self.emit_event(event_name="training_workflow_failure", payload={"error": error_msg, "exception": str(e)})
            raise WorkflowExecutionError(error_msg) from e
        finally:
            if distributed_utils.get_world_size() > 1:
                distributed_utils.cleanup_distributed_environment()