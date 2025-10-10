# cv_factory/shared_libs/orchestrators/cv_pipeline_factory.py (FINALIZED WITH DYNAMIC IMPORT)

import logging
import importlib # Standard library for dynamic module loading
from typing import Dict, Any, Type, Optional, Union, Callable, Type

# --- Import Core Orchestrators and Base Classes ---
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.orchestrators.cv_inference_orchestrator import CVInferenceOrchestrator
from shared_libs.orchestrators.cv_stream_inference_orchestrator import CVStreamInferenceOrchestrator
from shared_libs.inference.cv_predictor import CVPredictor 
from shared_libs.inference.base_cv_predictor import BaseCVPredictor 

# --- Import MLOps Services ---
from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger 
from shared_libs.ml_core.mlflow_service.implementations.mlflow_registry import MLflowRegistry 
from shared_libs.monitoring.event_emitter import ConsoleEventEmitter 
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry


logger = logging.getLogger(__name__)

class CVPipelineFactory:
    """
    A factory for creating the main orchestration pipelines (training, batch, or stream inference).

    This class centralizes the dependency management, using DYNAMIC IMPORT (importlib)
    to inject domain-specific logic (Postprocessor) into the Predictor.
    """
    _ORCHESTRATOR_MAP: Dict[str, Type[BaseOrchestrator]] = {
        "training": CVTrainingOrchestrator,
        "inference": CVInferenceOrchestrator,
        "stream_inference": CVStreamInferenceOrchestrator,
    }

    @classmethod
    def _create_mlops_services(cls) -> Dict[str, Any]:
        """Initializes common MLOps services required by all Orchestrators."""
        return {
            "logger_service": MLflowLogger(),
            "event_emitter": ConsoleEventEmitter(),
            "registry_service": MLflowRegistry(),
        }

    @classmethod
    def _get_class_from_string(cls, module_path: str, class_name: str) -> Type[Any]:
        """
        Dynamically imports a class object from a given module path and class name.
        """
        try:
            module = importlib.import_module(module_path)
            class_obj = getattr(module, class_name)
            return class_obj
        except Exception as e:
            logger.error(f"Failed dynamic import: Module '{module_path}' or class '{class_name}' not found.")
            # Wrap the error for clean upstream handling
            raise ImportError(f"Cannot load domain class dynamically. Check config module_path/class_name: {e}")

    @classmethod
    def _create_domain_postprocessor(cls, config: Dict[str, Any]) -> Any:
        """
        Creates the domain-specific Postprocessor instance using dynamic import based on config.
        """
        postprocessor_config = config.get('domain', {}).get('postprocessor', {})
        
        module_path = postprocessor_config.get('module_path')
        class_name = postprocessor_config.get('class_name')
        
        if not module_path or not class_name:
            raise ValueError("Configuration missing 'domain.postprocessor.module_path' or 'class_name' required for Dynamic Import.")

        # 1. Dynamically get the Class Object
        PostprocessorCls = cls._get_class_from_string(module_path, class_name)
        
        # 2. Instantiate the Class with configured parameters
        params = postprocessor_config.get('params', {})
        
        postprocessor_instance = PostprocessorCls(**params)
        
        logger.info(f"Dynamically loaded and instantiated Postprocessor: {class_name}")
        return postprocessor_instance

    @classmethod
    def _create_predictor(cls, config: Dict[str, Any], predictor_id: str) -> BaseCVPredictor:
        """
        Creates the CVPredictor instance, injecting the dynamically loaded Postprocessor.
        """
        
        # 1. Create Domain-specific Postprocessor (Dynamic Import and Instantiation)
        postprocessor_instance = cls._create_domain_postprocessor(config) 
        
        # 2. Create CVPredictor and Inject Postprocessor
        predictor_instance = CVPredictor(
            predictor_id=predictor_id,
            config=config,
            postprocessor=postprocessor_instance # <<< DI: Inject Dynamically Loaded Dependency >>>
        )
        
        return predictor_instance


    @classmethod
    def create(cls, config: Dict[str, Any], orchestrator_id: str) -> BaseOrchestrator:
        """
        Creates an orchestrator instance, injecting all necessary service dependencies.
        """
        pipeline_type = config.get('pipeline', {}).get('type', 'training').lower()
        orchestrator_cls = cls._ORCHESTRATOR_MAP.get(pipeline_type)
        
        if not orchestrator_cls:
            supported_types = ", ".join(cls._ORCHESTRATOR_MAP.keys())
            raise ValueError(f"Unsupported pipeline type: '{pipeline_type}'. Supported are: {supported_types}")
            
        # 1. Prepare MLOps Services (Dependencies)
        services = cls._create_mlops_services()
        
        # 2. Prepare Keyword Arguments for Orchestrator instantiation
        kwargs = {
            "orchestrator_id": orchestrator_id, 
            "config": config, 
            "logger_service": services['logger_service'],
            "event_emitter": services['event_emitter'],
        }
        
        # Add registry_service ONLY for Training Orchestrator
        if pipeline_type == "training":
             kwargs['registry_service'] = services['registry_service']

        # 3. Dependency Injection: Inject Predictor (Only required for Inference Orchestrators)
        if pipeline_type in ["inference", "stream_inference"]:
            predictor = cls._create_predictor(
                config=config,
                predictor_id=f"{orchestrator_id}-predictor"
            )
            # Inject the fully constructed Predictor into the Orchestrator
            kwargs['predictor'] = predictor 
        
        # 4. Instantiate Orchestrator
        try:
            return orchestrator_cls(**kwargs)
        except Exception as e:
            logger.critical(f"Failed to instantiate {pipeline_type} orchestrator (DI error): {e}")
            raise