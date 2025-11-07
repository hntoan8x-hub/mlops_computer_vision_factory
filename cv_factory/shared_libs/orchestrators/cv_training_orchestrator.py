# shared_libs/orchestrators/cv_training_orchestrator.py

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
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry 

# IMPORT MỚI: Deployment Orchestrator
from shared_libs.orchestrators.cv_deployment_orchestrator import CVDeploymentOrchestrator 

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
from shared_libs.ml_core.configs.orchestrator_config_schema import TrainingOrchestratorConfig 

logger = logging.getLogger(__name__)

class CVTrainingOrchestrator(BaseOrchestrator):
    """
    Orchestrates the end-to-end training and evaluation pipeline for a CV model, 
    integrating MLOps services and activating deployment upon successful registration.
    """
    
    def __init__(self, orchestrator_id: str, config: Dict[str, Any], logger_service: BaseTracker, event_emitter: Any, registry_service: BaseRegistry):
        
        # 1. Base Init (Validation, Logger/Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.registry = registry_service 

        # 2. Initialize Factories/Evaluator
        self.trainer_factory = TrainerFactory
        self.evaluation_orchestrator = EvaluationOrchestrator(
            config=self.config.get('evaluator', {}) 
        )
        # 3. Khởi tạo Deployment Orchestrator (Dependency Injection cho MLOps services)
        self.deployment_orchestrator = self._initialize_deployment_orchestrator()


    def _initialize_deployment_orchestrator(self) -> CVDeploymentOrchestrator:
        """
        Khởi tạo Deployment Orchestrator nếu cấu hình triển khai tồn tại.
        """
        deployment_config = self.config.get('deployment', {})
        if not deployment_config.get('enabled', True):
            logger.warning("Deployment is explicitly disabled in the configuration.")
            return None

        # Khởi tạo Deployment Orchestrator (sử dụng lại các services đã tiêm)
        return CVDeploymentOrchestrator(
            orchestrator_id=f"{self.orchestrator_id}-deploy",
            config=self.config, # Truyền config tổng thể
            logger_service=self.mlops_tracker, # Dịch vụ tracker
            event_emitter=self.emitter # Dịch vụ event
        )


    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        [IMPLEMENTED] Validates the configuration using the Pydantic schema.
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
        [IMPLEMENTED] Prepares CVDatasets and creates DDP-compatible DataLoaders.
        """
        self.logger.info(f"[{self.orchestrator_id}] Preparing data: Dataset -> DataLoader.")
        
        train_config = self.config['datasets']['train']
        val_config = self.config['datasets']['validation']
        
        common_dataset_params = {
            'data_connector_config': self.config['data_ingestion']['connector_config'],
            'preprocessing_config': self.config['preprocessing'],
            'labeling_config': self.config['labeling'] # Thêm labeling config
        }

        train_dataset = CVDataset(dataset_id="train_set", config=train_config, context='training', **common_dataset_params)
        val_dataset = CVDataset(dataset_id="validation_set", config=val_config, context='evaluation', **common_dataset_params)

        batch_size = self.config['trainer']['batch_size']
        num_workers = self.config['trainer'].get('num_workers', 4)
        is_distributed = distributed_utils.get_world_size() > 1
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(train_dataset, shuffle=True) if is_distributed else None,
            shuffle=True if not is_distributed else False,
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
        
        return train_loader, val_loader, val_loader # Dùng val_loader làm test_loader

    @measure_orchestrator_latency(orchestrator_type="CVTraining")
    def run(self) -> Tuple[Dict[str, Any], str]:
        """
        [IMPLEMENTED] Executes the full training workflow: Data -> Train -> Evaluate -> Register -> DEPLOY.
        """
        logged_model_uri = ""
        endpoint_id = "" 
        
        try:
            # BẮT ĐẦU MLFLOW RUN
            with self.mlops_tracker.start_run(run_name=self.orchestrator_id):
                
                self.log_metrics(self.config) 

                # 1. DATA PREPARATION (Giả định logic này đã được triển khai)
                train_loader, val_loader, test_loader = self._prepare_data()
                
                # 2. TRAINER INSTANTIATION (Giả định logic này đã được triển khai)
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
                self.log_metrics({f"final_{k}": v for k, v in final_metrics['metrics'].items()})
                
                # 5. MODEL REGISTRATION (Chỉ Rank 0 thực hiện)
                if distributed_utils.is_main_process():
                    # a. Save model artifact
                    logged_model_uri = self.mlops_tracker.log_model(trainer.model, artifact_path="model")
                    
                    # b. Register the model artifact
                    model_name = self.config['model'].get('name', 'cv_model')
                    model_version = self.registry.register_model(logged_model_uri, model_name)
                    
                    # c. TAGGING
                    git_sha = self.config.get('run_metadata', {}).get('git_sha', 'unknown')
                    config_hash = self.config.get('run_metadata', {}).get('config_hash', 'unknown')
                    self.registry.tag_model_version(
                        model_name, 
                        model_version, 
                        {'git_sha': git_sha, 'config_hash': config_hash, 'status': 'staging_ready'}
                    )
                    
                    logged_model_uri = f"models:/{model_name}/{model_version}"
                    self.logger.info(f"Model registered and tagged successfully: {logged_model_uri}")

                # BARRIER: Đồng bộ hóa URI mô hình đã đăng ký cho tất cả các ranks
                logged_model_uri = distributed_utils.broadcast_and_wait(logged_model_uri)
                
                # 6. TRIỂN KHAI MÔ HÌNH (CHỈ KÍCH HOẠT SAU KHI ĐĂNG KÝ)
                if self.deployment_orchestrator:
                    self.logger.info(f"[{self.orchestrator_id}] Kích hoạt Deployment Orchestrator...")
                    
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