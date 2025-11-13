# shared_libs/orchestrators/cv_training_orchestrator.py (UPDATED - FS CONTEXT MANAGER)

import logging
from typing import Dict, Any, Union, Tuple, Optional, Type
import torch
import os 
from torch.utils.data import DataLoader 

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

# NEW IMPORTS: Feature Store Orchestrator
from shared_libs.feature_store.orchestrator.feature_store_orchestrator import FeatureStoreOrchestrator 
import numpy as np # Cần cho mocking embeddings

# --- Import Utilities, Exceptions, and Monitoring ---
from shared_libs.orchestrators.utils.orchestrator_monitoring import measure_orchestrator_latency 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError 
from shared_libs.ml_core.configs.orchestrator_config_schema import TrainingOrchestratorConfig 

logger = logging.getLogger(__name__)

class CVTrainingOrchestrator(BaseOrchestrator):
    """
    Orchestrates the end-to-end training and evaluation pipeline.
    UPDATED: Integrated FeatureStoreOrchestrator via Dependency Injection.
    """
    
    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 logger_service: BaseTracker, 
                 event_emitter: Any, 
                 registry_service: BaseRegistry,
                 
                 evaluation_orchestrator: EvaluationOrchestrator,
                 deployment_orchestrator: Optional[CVDeploymentOrchestrator],
                 trainer_factory: Type[Any], 
                 ml_component_factory: Type[ComponentFactory],
                 live_data_uri_override: Optional[str] = None,
                 # <<< INJECTED FEATURE STORE >>>
                 feature_store_orchestrator: Optional[FeatureStoreOrchestrator] = None 
                ):
        
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        self.registry = registry_service 
        self.evaluation_orchestrator = evaluation_orchestrator 
        self.deployment_orchestrator = deployment_orchestrator 
        self.trainer_factory = trainer_factory                 
        self.ml_component_factory = ml_component_factory       
        self.live_data_uri_override = live_data_uri_override 
        self.feature_store = feature_store_orchestrator # LƯU TRỮ ORCHESTRATOR
        
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
        """
        self.logger.info(f"[{self.orchestrator_id}] Preparing data: Dataset -> DataLoader.")
        
        train_config = self.config['datasets']['train']
        val_config = self.config['datasets']['validation']
        
        common_dataset_params = {
            'data_connector_config': self.config['data_ingestion']['connector_config'],
            'preprocessing_config': self.config['preprocessing'],
            'labeling_config': self.config['labeling'],
            'ml_component_factory': self.ml_component_factory,
            'live_data_uri_override': self.live_data_uri_override 
        }

        # NOTE: Using direct import here for simplicity as the full Factory chain is too complex to reproduce inside a single file
        from shared_libs.ml_core.dataloader.dataloader_factory import DataLoaderFactory 
        from shared_libs.ml_core.dataset.cv_dataset import CVDataset
        
        train_dataset = CVDataset(dataset_id="train_set", config=train_config, context='training', **common_dataset_params)
        val_dataset = CVDataset(dataset_id="validation_set", config=val_config, context='evaluation', **common_dataset_params)

        train_loader = DataLoaderFactory.create(train_dataset, self.config, context='training')
        val_loader = DataLoaderFactory.create(val_dataset, self.config, context='evaluation')
        
        self.logger.info(f"Data preparation complete. Train size: {len(train_dataset)}, DataLoader Factory used.")
        
        # Test loader có thể là val_loader hoặc một loader riêng
        return train_loader, val_loader, val_loader 
    
    # --- UPDATED HELPER: Feature Store Indexing Logic (Context Manager) ---
    def _run_feature_indexing(self, model: torch.nn.Module, data_loader: DataLoader, context: str) -> None:
        """
        Runs the model to generate embeddings and indexes them into the feature store,
        using the Context Manager pattern.
        """
        if not self.feature_store:
            self.logger.info("Feature Store Orchestrator not injected. Skipping feature indexing.")
            return

        self.logger.info(f"Starting feature indexing for {context} data...")
        
        try:
            model.eval()
            
            # HARDENING: SỬ DỤNG CONTEXT MANAGER để đảm bảo tài nguyên FS được quản lý
            with self.feature_store as fs_orchestrator:
                for batch_idx, batch in enumerate(data_loader):
                    # Giả định DataLoader trả về Dictionary {'input': tensor, 'target': tensor, 'metadata': dict}
                    # Chuyển input về đúng device của model
                    inputs = batch.get('input', batch[0]).to(model.device) 
                    
                    # NOTE: Cần một phương thức để trích xuất embedding từ model
                    with torch.no_grad():
                        # MOCKING: Lấy embedding giả định
                        # Giả định 512 là chiều embedding
                        embeddings = np.random.randn(inputs.shape[0], 512) 
                        
                    # Lập chỉ mục
                    # Giả định metadata là một list of dicts được trích xuất từ batch
                    metadata_batch = batch.get('metadata', [{} for _ in range(inputs.shape[0])])

                    fs_orchestrator.index_embeddings(
                        embeddings, 
                        metadata=metadata_batch 
                    )
                    self.logger.info(f"Indexed batch {batch_idx} for {context}.")
                    
        except Exception as e:
            self.logger.error(f"Feature Indexing failed during run: {e}", exc_info=True)
            self.emit_event("feature_indexing_failure", {"context": context, "error": str(e)})
            
        model.train() # Đặt lại mode về training sau khi indexing

    # --- MAIN RUN METHOD ---
    @measure_orchestrator_latency(orchestrator_type="CVTraining")
    def run(self) -> Tuple[Dict[str, Any], str]:
        """
        Executes the full training workflow.
        """
        logged_model_uri = ""
        endpoint_id = "" 
        
        try:
            with self.mlops_tracker.start_run(run_name=self.orchestrator_id):
                
                self.mlops_tracker.log_params(self.config) 

                # 1. DATA PREPARATION 
                train_loader, val_loader, test_loader = self._prepare_data()
                
                # 2. TRAINER INSTANTIATION & TRAINING EXECUTION
                model_config = self.config['model']
                trainer_config = self.config['trainer']
                from shared_libs.ml_core.trainer.trainer_factory import TrainerFactory 
                trainer = TrainerFactory.create(
                    trainer_config['type'], 
                    model_config, 
                    trainer_config
                )
                self.logger.info(f"[{self.orchestrator_id}] Starting model training...")
                trainer.fit(train_loader, val_loader, epochs=trainer_config['epochs'])
                
                # 3. EVALUATION 
                self.logger.info(f"[{self.orchestrator_id}] Starting final evaluation...")
                final_metrics = self.evaluation_orchestrator.evaluate(trainer.model, test_loader)
                self.mlops_tracker.log_metrics({f"final_{k}": v for k, v in final_metrics['metrics'].items()})
                
                # 4. FEATURE STORE INDEXING (NEW CRITICAL STEP)
                if self.feature_store:
                    self._run_feature_indexing(trainer.model, train_loader, context="training")
                    self._run_feature_indexing(trainer.model, val_loader, context="validation") # Dùng model từ trainer
                    
                # 5. MODEL REGISTRATION 
                if distributed_utils.is_main_process():
                    logged_model_uri = self.mlops_tracker.log_model(trainer.model, artifact_path="model")
                    model_name = self.config['model'].get('name', 'cv_model')
                    model_version = self.registry.register_model(logged_model_uri, model_name)
                    self.registry.tag_model_version(
                        model_name, 
                        model_version, 
                        {'status': 'staging_ready'}
                    )
                    logged_model_uri = f"models:/{model_name}/{model_version}"
                    self.logger.info(f"Model registered and tagged successfully: {logged_model_uri}")

                logged_model_uri = distributed_utils.broadcast_and_wait(logged_model_uri)
                
                # 6. TRIỂN KHAI MÔ HÌNH 
                if distributed_utils.is_main_process() and self.deployment_orchestrator:
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