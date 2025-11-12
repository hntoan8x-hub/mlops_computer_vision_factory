# shared_libs/ml_core/mlflow_service/factories/registry_factory.py (HARDENED - DYNAMIC IMPORT)

import logging
import importlib 
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
# LOẠI BỎ: from shared_libs.ml_core.mlflow_service.implementations.mlflow_registry import MLflowRegistry
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceException

logger = logging.getLogger(__name__)

class RegistryFactory:
    """
    A factory class for creating different types of model registries.

    HARDENED: Now uses dynamic import to load implementation classes, enabling 
    addition of new registry types without factory code modification.
    """
    # LOẠI BỎ: _REGISTRY_MAP tĩnh

    @staticmethod
    def _get_class_from_config(registry_type: str, config: Dict[str, Any]) -> Type[BaseRegistry]:
        """
        Dynamically imports the concrete Registry class based on configuration.
        """
        registry_config = config.get(registry_type.lower(), {})
        module_path = registry_config.get('module_path')
        class_name = registry_config.get('class_name')

        if not module_path or not class_name:
            raise MLflowServiceException(
                f"Dynamic configuration for registry '{registry_type}' is missing 'module_path' or 'class_name'."
            )

        try:
            module = importlib.import_module(module_path)
            registry_cls = getattr(module, class_name)
            if not issubclass(registry_cls, BaseRegistry):
                 raise TypeError(f"Class '{class_name}' must inherit from BaseRegistry.")
            return registry_cls
            
        except ImportError as e:
            logger.error(f"Module import failed for registry '{registry_type}': {e}")
            raise MLflowServiceException(f"Failed to import registry class: {e}")
        except AttributeError as e:
            logger.error(f"Class '{class_name}' not found in module '{module_path}'.")
            raise MLflowServiceException(f"Registry class not found: {e}")


    @classmethod
    def create(cls, registry_type: str, config: Optional[Dict[str, Any]] = None) -> BaseRegistry:
        """
        Creates and returns a registry instance based on its type.

        Args:
            registry_type (str): The type of registry to create (e.g., "mlflow").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseRegistry: An instance of the requested registry.

        Raises:
            MLflowServiceException: If the specified registry_type is not supported or instantiation fails.
        """
        config = config or {}

        # 1. Dynamically find the class
        registry_cls = cls._get_class_from_config(registry_type, config)
        
        # 2. Extract constructor parameters
        constructor_params = config.get('params', {}) 

        logger.info(f"Creating instance of {registry_cls.__name__} using dynamic configuration...")
        try:
            return registry_cls(**constructor_params)
        except Exception as e:
            logger.error(f"Failed to instantiate registry '{registry_type}' with params {constructor_params}: {e}")
            raise MLflowServiceException(f"Registry creation failed for type '{registry_type}': {e}")