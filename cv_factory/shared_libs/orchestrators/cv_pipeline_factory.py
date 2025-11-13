# shared_libs/orchestrators/cv_pipeline_factory.py (UPDATED - ADDING TMR GLUE)

import logging
import importlib 
from typing import Dict, Any, Type, Optional, Union, Callable, Tuple

# --- Import Core Orchestrators and Base Classes ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.orchestrators.cv_inference_orchestrator import CVInferenceOrchestrator
from shared_libs.orchestrators.cv_stream_inference_orchestrator import CVStreamInferenceOrchestrator
from shared_libs.orchestrators.cv_deployment_orchestrator import CVDeploymentOrchestrator 
from shared_libs.inference.cv_predictor import CVPredictor 
from shared_libs.inference.base_cv_predictor import BaseCVPredictor 

# --- Import TMR Components (NEW IMPORTS) ---
from shared_libs.ml_core.monitoring.orchestrator.monitoring_orchestrator import MonitoringOrchestrator
from shared_libs.ml_core.retraining.retrain_factory import RetrainOrchestratorFactory 
from shared_libs.ml_core.retraining.tmr_facade import TMRFacade 
from shared_libs.ml_core.retraining.orchestrator.retrain_orchestrator import RetrainOrchestrator 

# --- Import MLOps Services & Contracts ---
from shared_libs.ml_core.mlflow_service.mlflow_service import MLflowService 
from shared_libs.ml_core.mlflow_service.configs.mlflow_config_schema import MLflowConfig 
from shared_libs.ml_core.monitoring.event_emitter import ConsoleEventEmitter 
from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator 
from shared_libs.ml_core.trainer.trainer_factory import TrainerFactory 
from shared_libs.ml_core.pipeline_components_cv.factories.component_factory import ComponentFactory

# --- Import Deployment Factories/Contracts ---
from shared_libs.deployment.factory.deployer_factory import DeployerFactory 
from shared_libs.deployment.factory.traffic_controller_factory import TrafficControllerFactory 
from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.deployment.contracts.base_traffic_controller import BaseTrafficController

# --- Import Model Loading Service ---
from shared_libs.ml_core.model_loading.base_model_loading_service import BaseModelLoadingService
from shared_libs.ml_core.model_loading.mlflow_model_loading_service import MLflowModelLoadingService

# --- Feature Store Imports ---
from shared_libs.feature_store.orchestrator.feature_store_orchestrator import FeatureStoreOrchestrator 
from shared_libs.feature_store.factories.vector_store_factory import VectorStoreFactory 
from shared_libs.feature_store.factories.retriever_factory import RetrieverFactory 
from shared_libs.ml_core.configs.feature_store_config_schema import FeatureStoreConfig 


logger = logging.getLogger(__name__)

class CVPipelineFactory:
    """
    A factory for creating the main orchestration pipelines (Composition Root).
    """
    _ORCHESTRATOR_MAP: Dict[str, Type[Any]] = { # Changed Type[BaseOrchestrator] to Type[Any] to include TMRFacade
        "training": CVTrainingOrchestrator,
        "inference": CVInferenceOrchestrator,
        "stream_inference": CVStreamInferenceOrchestrator,
        "deployment": CVDeploymentOrchestrator,
        "retrain_check": TMRFacade, # <<< NEW: TMR Façade là Orchestrator chính cho luồng kiểm tra
    }

    @classmethod
    def _create_mlops_services(cls, mlflow_config: MLflowConfig) -> Dict[str, Any]:
        """Initializes common MLOps services by creating the MLflowService Façade."""
        mlflow_service_facade = MLflowService(config=mlflow_config)
        
        return {
            "mlflow_service_facade": mlflow_service_facade,
            "logger_service": mlflow_service_facade.tracker, 
            "registry_service": mlflow_service_facade.registry, 
            "event_emitter": ConsoleEventEmitter(),
        }

    @classmethod
    def _create_model_loading_service(cls, config: Dict[str, Any]) -> BaseModelLoadingService:
        """Initializes the Model Loading Service (MLflowModelLoadingService)."""
        loading_config = config.get('model_loading', {})
        return MLflowModelLoadingService(
            service_id="mlflow-loader",
            config=loading_config
        )

    # --- NEW HELPER: Feature Store Assembly ---
    @classmethod
    def _create_feature_store_orchestrator(cls, config: Dict[str, Any]) -> Optional[FeatureStoreOrchestrator]:
        """
        Assembles and returns the FeatureStoreOrchestrator instance.
        """
        fs_config_raw = config.get('feature_store')
        if not fs_config_raw or not fs_config_raw.get('enabled', False):
            logger.info("Feature Store is disabled in config. Skipping initialization.")
            return None

        # 1. Validate Config (CRITICAL)
        try:
            fs_config = FeatureStoreConfig(**fs_config_raw) 
        except Exception as e:
            logger.error(f"Feature Store Configuration validation failed: {e}")
            raise ValueError(f"Invalid Feature Store Config: {e}")

        # 2. Tạo Vector Store (Connector/I/O)
        vs_config = fs_config.vector_store
        vector_store = VectorStoreFactory.create(
            connector_type=vs_config.type,
            config=vs_config.connection_params 
        )

        # 3. Tạo Retriever (Logic) và tiêm Vector Store (DI)
        retriever_config = fs_config.retriever
        dependencies = {"vector_store": vector_store}
        
        retriever = RetrieverFactory.create(
            retriever_type=retriever_config.type,
            dependencies=dependencies,
            config=retriever_config.params or {}
        )

        # 4. Tạo Orchestrator Façade 
        return FeatureStoreOrchestrator(config=fs_config_raw)


    # --- Utility Methods ( giữ nguyên ) ---
    @staticmethod
    def _get_class_from_string(module_path: str, class_name: str) -> Type[Any]:
        """Dynamically imports a class object from a given module path and class name."""
        try:
            module = importlib.import_module(module_path)
            class_obj = getattr(module, class_name)
            return class_obj
        except Exception as e:
            logger.error(f"Failed dynamic import: Module '{module_path}' or class '{class_name}' not found.")
            raise ImportError(f"Cannot load domain class dynamically. Check config module_path/class_name: {e}")

    @classmethod
    def _create_domain_postprocessor(cls, config: Dict[str, Any]) -> Any:
        """Creates the domain-specific Postprocessor instance using dynamic import."""
        postprocessor_config = config.get('domain', {}).get('postprocessor', {})
        module_path = postprocessor_config.get('module_path')
        class_name = postprocessor_config.get('class_name')
        
        if not module_path or not class_name:
            raise ValueError("Configuration missing 'domain.postprocessor.module_path' or 'class_name'.")
        PostprocessorCls = cls._get_class_from_string(module_path, class_name)
        params = postprocessor_config.get('params', {})
        postprocessor_instance = PostprocessorCls(**params)
        logger.info(f"Dynamically loaded and instantiated Postprocessor: {class_name}")
        return postprocessor_instance


    @classmethod
    def _create_predictor(cls, config: Dict[str, Any], predictor_id: str, model_uri: Optional[str] = None) -> BaseCVPredictor:
        """
        Creates the CVPredictor instance by delegating model loading 
        and injecting the loaded model AND FeatureStoreOrchestrator.
        """
        loaded_model = None
        if model_uri:
             # 1a. Tải Model
             loader_service = cls._create_model_loading_service(config)
             device = config.get('model', {}).get('device')
             loaded_model = loader_service.load(model_uri, target_device=device)
        
        # 1b. Tạo Feature Store (Nếu cần)
        feature_store_orchestrator = cls._create_feature_store_orchestrator(config)

        # 2. Tạo Domain-specific Postprocessor
        postprocessor_instance = cls._create_domain_postprocessor(config) 
        
        # 3. Tạo CVPredictor và TIÊM Model + Feature Store
        predictor_instance = CVPredictor(
            predictor_id=predictor_id,
            config=config,
            postprocessor=postprocessor_instance,
            loaded_model=loaded_model,
            feature_store=feature_store_orchestrator # <<< TIÊM FEATURE STORE >>>
        )
        
        return predictor_instance


    @classmethod
    def _create_deployment_services(cls, config: Dict[str, Any]) -> Dict[str, Union[BaseDeployer, Optional[BaseTrafficController]]]:
        """
        CẬP NHẬT: Tạo và trả về các instance Deployer và Traffic Controller (Hardened DI).
        """
        
        deployment_config = config.get('deployment', {})
        # Sử dụng 'kubernetes' làm default cho AI Factory
        platform_type = deployment_config.get('platform_type', 'kubernetes') 
        endpoint_name = deployment_config.get('endpoint_name') # CRITICAL: Phải có trong config
        
        if not endpoint_name:
             logger.error("Deployment configuration missing 'endpoint_name'. Cannot create deployment service.")
             raise ValueError("Deployment config requires 'endpoint_name'.")
        
        # 1. Khởi tạo Deployer (Sử dụng Factory củng cố)
        platform_config = deployment_config.get(platform_type, {}) 
        
        deployer = DeployerFactory.create_deployer( 
            platform_type=platform_type,
            config=platform_config 
        )

        traffic_controller = None
        traffic_config_raw = deployment_config.get('traffic_controller', {})

        # 2. Khởi tạo Traffic Controller (Sử dụng Factory củng cố)
        if traffic_config_raw.get('enabled', False):
            controller_type = traffic_config_raw.get('type', 'istio')
            
            # Lấy tham số cho Traffic Controller
            controller_params = traffic_config_raw.get('params', {})
            
            traffic_controller = TrafficControllerFactory.create_controller( 
                controller_type=controller_type,
                endpoint_name=endpoint_name, # <-- TIÊM TÊN ENDPOINT
                config=controller_params 
            )
        
        return {
            "deployer": deployer,
            "traffic_controller": traffic_controller
        }


    @classmethod
    def _create_training_dependencies(cls, config: Dict[str, Any], services: Dict[str, Any], orchestrator_id: str) -> Dict[str, Any]:
        """
        Handles complex dependency instantiation for the Training Orchestrator.
        """
        
        evaluation_orchestrator = EvaluationOrchestrator(
            config=config.get('evaluator', {}) 
        )
        
        deployment_config = config.get('deployment', {})
        deployment_orchestrator = None
        
        if deployment_config.get('enabled', True):
            deploy_services = cls._create_deployment_services(config)
            
            # INJECTION OF REGISTRY SERVICE FOR CVDeploymentOrchestrator (HARDENING STEP 2)
            deployment_orchestrator = CVDeploymentOrchestrator(
                orchestrator_id=f"{orchestrator_id}-deploy",
                config=config, 
                logger_service=services['logger_service'], 
                event_emitter=services['event_emitter'],
                deployer=deploy_services['deployer'],
                traffic_controller=deploy_services['traffic_controller'],
                registry_service=services['registry_service'] # <<< HARDENING: INJECT REGISTRY >>>
            )

        # <<< FEATURE STORE ASSEMBLY AND INJECTION >>>
        feature_store_orchestrator = cls._create_feature_store_orchestrator(config)

        return {
            "evaluation_orchestrator": evaluation_orchestrator,
            "deployment_orchestrator": deployment_orchestrator,
            "trainer_factory": TrainerFactory,
            "ml_component_factory": ComponentFactory,
            "feature_store_orchestrator": feature_store_orchestrator # <<< INJECT FEATURE STORE >>>
        }
    
    @classmethod
    def _create_monitoring_orchestrator(cls, config: Dict[str, Any], services: Dict[str, Any]) -> MonitoringOrchestrator:
        """NEW HELPER: Creates the Monitoring Orchestrator."""
        return MonitoringOrchestrator(
            config=config.get('monitoring', {}),
            event_emitter=services['event_emitter'] # <<< INJECT EVENT EMITTER >>>
        )

    @classmethod
    def _create_retraining_orchestrator(cls, config: Dict[str, Any], services: Dict[str, Any]) -> RetrainOrchestrator:
        """NEW HELPER: Creates the Retraining Orchestrator."""
        return RetrainOrchestratorFactory.create(
            config=config.get('retraining', {}),
            logger_service=services['logger_service'],
            registry_service=services['registry_service'],
            event_emitter=services['event_emitter']
        )

    @classmethod
    def _create_tmr_dependencies(cls, config: Dict[str, Any], services: Dict[str, Any]) -> Dict[str, Any]:
        """NEW HELPER: Creates dependencies for the TMRFacade."""
        
        monitoring_orchestrator = cls._create_monitoring_orchestrator(config, services)
        retrain_orchestrator = cls._create_retraining_orchestrator(config, services)

        return {
            "monitoring_orchestrator": monitoring_orchestrator,
            "retrain_orchestrator": retrain_orchestrator
        }


    @classmethod
    def create(cls, config: Dict[str, Any], orchestrator_id: str, pipeline_type: Optional[str] = None) -> Any: # Type[Any] to include TMRFacade
        """
        Creates an orchestrator instance, performing all required dependency injection 
        based on the pipeline type.
        """
        pipeline_type = pipeline_type or config.get('pipeline', {}).get('type', 'training').lower()
        orchestrator_cls = cls._ORCHESTRATOR_MAP.get(pipeline_type)
        
        if not orchestrator_cls:
            supported_types = ", ".join(cls._ORCHESTRATOR_MAP.keys())
            raise ValueError(f"Unsupported pipeline type: '{pipeline_type}'. Supported are: {supported_types}")
            
        # Validate MLOps config
        try:
             mlflow_config_obj = MLflowConfig(**config.get('mlflow', {}))
        except Exception as e:
             logger.critical(f"MLflow Configuration validation failed: {e}")
             raise

        # 1. Prepare MLOps Services
        services = cls._create_mlops_services(mlflow_config_obj)
        
        # 2. Prepare Base Keyword Arguments 
        kwargs = {
            "orchestrator_id": orchestrator_id, 
            "config": config, 
            "logger_service": services['logger_service'], 
            "event_emitter": services['event_emitter'],
        }
        
        # 3. Dependency Injection: Context-Specific Dependencies
        if pipeline_type == "training":
             kwargs['registry_service'] = services['registry_service']
             training_dependencies = cls._create_training_dependencies(config, services, orchestrator_id)
             kwargs.update(training_dependencies)

        elif pipeline_type in ["inference", "stream_inference"]:
            model_uri = config.get('model', {}).get('uri')
            if not model_uri:
                 raise ValueError(f"Inference pipeline requires 'model.uri' in configuration.")
                 
            predictor = cls._create_predictor(
                config=config, 
                predictor_id=f"{orchestrator_id}-predictor",
                model_uri=model_uri 
            )
            kwargs['predictor'] = predictor
        
        elif pipeline_type == "deployment":
            kwargs['registry_service'] = services['registry_service']
            deploy_services = cls._create_deployment_services(config)
            kwargs.update(deploy_services)
            
        elif pipeline_type == "retrain_check": # <<< NEW: TMR FACADE DEPENDENCY >>>
            tmr_dependencies = cls._create_tmr_dependencies(config, services)
            # TMRFacade không nhận logger_service/event_emitter trực tiếp, mà nhận các orchestrator đã được tiêm
            kwargs = tmr_dependencies


        # 4. Instantiate Orchestrator
        try:
            return orchestrator_cls(**kwargs)
        except Exception as e:
            logger.critical(f"Failed to instantiate {pipeline_type} orchestrator (DI error): {e}")
            raise