import logging
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry
from shared_libs.ml_core.mlflow_service.implementations.mlflow_registry import MLflowRegistry
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowServiceException

logger = logging.getLogger(__name__)

class RegistryFactory:
    """
    A factory class for creating different types of model registries.

    This class centralizes the creation logic, allowing for a config-driven
    approach to model versioning.
    """
    _REGISTRY_MAP: Dict[str, Type[BaseRegistry]] = {
        "mlflow": MLflowRegistry,
        # Add other registries here
    }

    @classmethod
    def create(cls, registry_type: str, config: Optional[Dict[str, Any]] = None) -> BaseRegistry:
        """
        Creates and returns a registry instance based on its type.

        Args:
            registry_type (str): The type of registry to create (e.g., "mlflow").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the registry's constructor.

        Returns:
            BaseRegistry: An instance of the requested registry.

        Raises:
            MLflowServiceException: If the specified registry_type is not supported.
        """
        config = config or {}
        registry_cls = cls._REGISTRY_MAP.get(registry_type.lower())
        
        if not registry_cls:
            supported_registries = ", ".join(cls._REGISTRY_MAP.keys())
            logger.error(f"Unsupported registry type: '{registry_type}'. Supported types are: {supported_registries}")
            raise MLflowServiceException(f"Unsupported registry type: '{registry_type}'. Supported types are: {supported_registries}")

        logger.info(f"Creating instance of {registry_cls.__name__}...")
        try:
            return registry_cls(**config)
        except Exception as e:
            logger.error(f"Failed to instantiate registry '{registry_type}' with config {config}: {e}")
            raise