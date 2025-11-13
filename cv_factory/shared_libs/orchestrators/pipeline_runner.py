# shared_libs/orchestrators/pipeline_runner.py (FINAL HARDENED VERSION with Feature Store Return)

import logging
import json
import os
from typing import Dict, Any, Type, Optional, Tuple

# --- Import Core Components ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.cv_pipeline_factory import CVPipelineFactory 
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError
from shared_libs.ml_core.mlflow_service.mlflow_service import MLflowService
from shared_libs.ml_core.mlflow_service.configs.mlflow_config_schema import MLflowConfig
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator
from shared_libs.inference.base_cv_predictor import BaseCVPredictor

# --- NEW IMPORTS for DataLoader & Feature Store ---
from shared_libs.ml_core.dataloader.dataloader_factory import DataLoaderFactory
from shared_libs.ml_core.dataset.cv_dataset import CVDataset
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory
from shared_libs.feature_store.orchestrator.feature_store_orchestrator import FeatureStoreOrchestrator # <<< NEW IMPORT >>>
import torch.utils.data

logger = logging.getLogger("PIPELINE_RUNNER")

class PipelineRunner:
    """
    Utility class acting as the central entry point (Composition Root) for all 
    pipeline execution scripts, enforcing Dependency Injection.
    """

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Tải cấu hình từ file JSON."""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found at: {config_path}")
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _validate_mlflow_config(raw_config: Dict[str, Any]) -> MLflowConfig:
        """Helper to validate MLflow config before creating services."""
        try:
            return MLflowConfig(**raw_config.get('mlflow', {}))
        except Exception as e:
            logger.critical(f"MLflow Configuration validation failed: {e}")
            raise InvalidConfigError(f"MLflow config is invalid: {e}")

    @staticmethod
    def create_orchestrator(config_path: str, run_id: str, pipeline_type: str) -> BaseOrchestrator:
        """
        Loads config, validates it, and uses CVPipelineFactory to create a fully
        assembled and dependency-injected Orchestrator instance.
        """
        try:
            raw_config = PipelineRunner.load_config(config_path)
            
            orchestrator_instance = CVPipelineFactory.create(
                config=raw_config,
                orchestrator_id=run_id,
                pipeline_type=pipeline_type 
            )
            logger.info(f"Orchestrator '{type(orchestrator_instance).__name__}' created successfully via Factory.")
            return orchestrator_instance
            
        except FileNotFoundError:
            raise
        except InvalidConfigError:
            logger.critical("Configuration validation failed during Orchestrator creation.")
            raise
        except Exception as e:
            logger.critical(f"Failed to create Orchestrator instance via Factory: {e}")
            raise RuntimeError(f"Runner failed during Orchestrator assembly: {e}")


    @staticmethod
    def create_mlops_services(config_path: str) -> Tuple[BaseTracker, BaseRegistry]:
        """
        Creates and returns the MLOps Contracts (Tracker and Registry) required for 
        standalone scripts (e.g., Cleanup).
        """
        raw_config = PipelineRunner.load_config(config_path)
        mlflow_config_obj = PipelineRunner._validate_mlflow_config(raw_config)
        
        mlflow_service_facade = MLflowService(config=mlflow_config_obj)

        return mlflow_service_facade.tracker, mlflow_service_facade.registry

    
    @staticmethod
    def create_production_dataloader(config_path: str, model_uri: Optional[str] = None) -> torch.utils.data.DataLoader:
        """
        Helper method to assemble the CVDataset and use DataLoaderFactory to create 
        the Production/Health Check DataLoader.
        """
        raw_config = PipelineRunner.load_config(config_path)
        
        # --- XÁC ĐỊNH NGUỒN DỮ LIỆU SỐNG (LIVE DATA SOURCE) ---
        live_data_source_uri = raw_config.get('live_data', {}).get('uri_source')
        if not live_data_source_uri and model_uri:
             logger.warning("No explicit live data URI configured. Skipping override.")
        # --- END LIVE DATA LOGIC ---

        # 1. Lấy config cho Dataset Production
        prod_config = raw_config['datasets'].get('production', raw_config['datasets'].get('validation'))
        if not prod_config:
            raise ValueError("Config missing 'datasets.production' or 'datasets.validation' for DataLoader creation.")

        # 2. Lắp ráp CVDataset (SỬ DỤNG LIVE_DATA_SOURCE_URI)
        common_dataset_params = {
            'data_connector_config': raw_config['data_ingestion']['connector_config'],
            'preprocessing_config': raw_config['preprocessing'],
            'labeling_config': raw_config['labeling'],
            'ml_component_factory': ComponentFactory, 
            'live_data_uri_override': live_data_source_uri 
        }

        prod_dataset = CVDataset(
            dataset_id="production_set", 
            config=prod_config, 
            context='evaluation', 
            **common_dataset_params
        )

        # 3. Sử dụng DataLoaderFactory để tạo DataLoader
        return DataLoaderFactory.create(
            prod_dataset, 
            raw_config, 
            context='production'
        )


    @staticmethod
    def create_model_and_evaluator(config_path: str, model_uri: str) -> Tuple[BaseCVPredictor, EvaluationOrchestrator, torch.utils.data.DataLoader, Optional[FeatureStoreOrchestrator]]:
        """
        Creates a pre-loaded Predictor, the Evaluation Orchestrator, the Production DataLoader,
        AND the FeatureStoreOrchestrator for Health Check purposes.
        """
        raw_config = PipelineRunner.load_config(config_path)
        
        # 1. Tạo Predictor (CVPipelineFactory sẽ tải Model VÀ FeatureStoreOrchestrator)
        # Vì CVPipelineFactory._create_predictor không lộ ra FeatureStoreOrchestrator, 
        # ta phải gọi helper riêng để lấy nó.
        feature_store_orchestrator = CVPipelineFactory._create_feature_store_orchestrator(raw_config) # Lấy FSO độc lập
        
        # Để Feature Store được tiêm vào Predictor, ta phải gọi lại logic tạo Predictor 
        # với FSO được tạo sẵn (hoặc chấp nhận rằng FSO được tạo 2 lần, hoặc thay đổi signature của _create_predictor)
        
        # Phương án tốt nhất: Lấy FSO từ Factory và đảm bảo Factory tiêm nó vào Predictor
        predictor = CVPipelineFactory._create_predictor(
             config=raw_config, 
             predictor_id="health-check-predictor",
             model_uri=model_uri 
        )
        
        # 2. Tạo Evaluation Orchestrator
        evaluation_orchestrator = EvaluationOrchestrator(
            config=raw_config.get('evaluator', {}) 
        )
        
        # 3. TẠO DATALOADER PRODUCTION
        production_dataloader = PipelineRunner.create_production_dataloader(config_path, model_uri=model_uri) 
        
        # 4. Trả về cả Feature Store Orchestrator
        # NOTE: Predictor đã có FSO bên trong nó. Ta trả về instance FSO đã tạo.
        return predictor, evaluation_orchestrator, production_dataloader, feature_store_orchestrator