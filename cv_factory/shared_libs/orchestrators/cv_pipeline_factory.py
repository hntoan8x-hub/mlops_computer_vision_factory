# shared_libs/orchestrators/cv_pipeline_factory.py (FINALIZED WITH MODEL LOADING SERVICE DI)

import logging
import importlib 
from typing import Dict, Any, Type, Optional, Union, Callable, Type, Tuple

# --- Import Core Orchestrators and Base Classes ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.orchestrators.cv_inference_orchestrator import CVInferenceOrchestrator
from shared_libs.orchestrators.cv_stream_inference_orchestrator import CVStreamInferenceOrchestrator
from shared_libs.orchestrators.cv_deployment_orchestrator import CVDeploymentOrchestrator 
from shared_libs.inference.cv_predictor import CVPredictor 
from shared_libs.inference.base_cv_predictor import BaseCVPredictor 

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

# --- NEW IMPORTS: Model Loading Service ---
from shared_libs.ml_core.model_loading.base_model_loading_service import BaseModelLoadingService
from shared_libs.ml_core.model_loading.mlflow_model_loading_service import MLflowModelLoadingService


logger = logging.getLogger(__name__)

class CVPipelineFactory:
    """
    A factory for creating the main orchestration pipelines (training, inference, stream, or deployment).

    This class centralizes the dependency management and acts as the Composition Root, 
    performing all necessary Dependency Injection using specialized services (like Model Loading).
    """
    _ORCHESTRATOR_MAP: Dict[str, Type[BaseOrchestrator]] = {
        "training": CVTrainingOrchestrator,
        "inference": CVInferenceOrchestrator,
        "stream_inference": CVStreamInferenceOrchestrator,
        "deployment": CVDeploymentOrchestrator,
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
        # Tạm thời hardcode MLflow. Trong kiến trúc đầy đủ, đây sẽ là một Factory.
        return MLflowModelLoadingService(
            service_id="mlflow-loader",
            config=loading_config
        )

    # --- Utility Methods ---
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
        to the ModelLoadingService and injecting the loaded model.
        """
        loaded_model = None
        if model_uri:
             # 1. Tải Model (Chuyển trách nhiệm)
             loader_service = cls._create_model_loading_service(config)
             device = config.get('model', {}).get('device')
             loaded_model = loader_service.load(model_uri, target_device=device)
        
        # 2. Tạo Domain-specific Postprocessor
        postprocessor_instance = cls._create_domain_postprocessor(config) 
        
        # 3. Tạo CVPredictor và TIÊM Model đã tải (DI)
        predictor_instance = CVPredictor(
            predictor_id=predictor_id,
            config=config,
            postprocessor=postprocessor_instance,
            loaded_model=loaded_model # <<< TIÊM MODEL ĐÃ TẢI >>>
        )
        
        return predictor_instance


    @classmethod
    def _create_deployment_services(cls, config: Dict[str, Any]) -> Dict[str, Union[BaseDeployer, Optional[BaseTrafficController]]]:
        """Creates and returns the Deployer and Traffic Controller instances."""
        deployment_config = config.get('deployment', {})
        platform = deployment_config.get('platform', 'sagemaker')
        
        deployer: BaseDeployer = DeployerFactory.create(
            deployer_type=platform,
            config=deployment_config.get('config', {}) 
        )

        traffic_controller: Optional[BaseTrafficController] = None
        traffic_config = deployment_config.get('config', {}).get('traffic_controller', {})

        if traffic_config.get('enabled', False):
            traffic_controller = TrafficControllerFactory.create(
                controller_type=traffic_config.get('type', 'istio'),
                config=traffic_config.get('params', {})
            )
        
        return {
            "deployer": deployer,
            "traffic_controller": traffic_controller
        }


    @classmethod
    def _create_training_dependencies(cls, config: Dict[str, Any], services: Dict[str, Any], orchestrator_id: str) -> Dict[str, Any]:
        """
        Handles complex dependency instantiation for the Training Orchestrator, 
        including nested deployment service injection.
        """
        
        evaluation_orchestrator = EvaluationOrchestrator(
            config=config.get('evaluator', {}) 
        )
        
        deployment_config = config.get('deployment', {})
        deployment_orchestrator = None
        
        if deployment_config.get('enabled', True):
            deploy_services = cls._create_deployment_services(config)
            
            deployment_orchestrator = CVDeploymentOrchestrator(
                orchestrator_id=f"{orchestrator_id}-deploy",
                config=config, 
                logger_service=services['logger_service'], 
                event_emitter=services['event_emitter'],
                deployer=deploy_services['deployer'],
                traffic_controller=deploy_services['traffic_controller']
            )

        return {
            "evaluation_orchestrator": evaluation_orchestrator,
            "deployment_orchestrator": deployment_orchestrator,
            "trainer_factory": TrainerFactory,
            "ml_component_factory": ComponentFactory
        }


    @classmethod
    def create(cls, config: Dict[str, Any], orchestrator_id: str) -> BaseOrchestrator:
        """
        Creates an orchestrator instance, performing all required dependency injection 
        based on the pipeline type.
        """
        pipeline_type = config.get('pipeline', {}).get('type', 'training').lower()
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
                model_uri=model_uri # Tải model ngay tại Composition Root
            )
            kwargs['predictor'] = predictor 
        
        elif pipeline_type == "deployment":
            kwargs['registry_service'] = services['registry_service'] 
            deploy_services = cls._create_deployment_services(config)
            kwargs.update(deploy_services) 

        # 4. Instantiate Orchestrator
        try:
            return orchestrator_cls(**kwargs)
        except Exception as e:
            logger.critical(f"Failed to instantiate {pipeline_type} orchestrator (DI error): {e}")
            raise