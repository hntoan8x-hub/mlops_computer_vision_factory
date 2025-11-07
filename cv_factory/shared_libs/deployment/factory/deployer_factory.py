# shared_libs/deployment/factory/deployer_factory.py

import logging
from typing import Dict, Any, Type, Literal
from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.ml_core.trainer.exceptions import UnsupportedModelError 

# --- Import Concrete Cloud Adapters ---
# Hardening 1: Import all concrete implementations
from infra_deployment.cloud.aws_sagemaker_deploy import AWSSageMakerDeployer
from infra_deployment.cloud.gcp_vertex_deploy import GCPVertexDeployer
from infra_deployment.cloud.azure_ml_deploy import AzureMLDeployer
from shared_libs.deployment.implementations.kubernetes_deployer import KubernetesDeployer
from shared_libs.deployment.implementations.on_premise_deployer import OnPremiseDeployer 

logger = logging.getLogger(__name__)


# Hardening 2: Registry hoàn chỉnh
DEPLOYER_REGISTRY: Dict[str, Type[Any]] = { # Sử dụng Any để linh hoạt với các classes Facade
    "aws": AWSSageMakerDeployer,
    "gcp": GCPVertexDeployer,
    "azure": AzureMLDeployer,
    "kubernetes": KubernetesDeployer, 
    "on_premise": OnPremiseDeployer
}

class DeployerFactory:
    """
    Factory for creating concrete Deployment Adapters based on the configuration.
    This enforces Inversion of Control for the infrastructure layer.
    """

    @staticmethod
    def create_deployer(platform_type: str, config: Dict[str, Any]) -> Any:
        """
        Hardening 3: Creates and returns an initialized Deployer instance.
        """
        provider_key = platform_type.lower()
        
        DeployerClass = DEPLOYER_REGISTRY.get(provider_key)
        
        if DeployerClass is None:
            available_keys = list(DEPLOYER_REGISTRY.keys())
            raise UnsupportedModelError(
                f"Deployment provider '{platform_type}' is not supported. Available: {available_keys}"
            )
        
        logger.info(f"🏭 Creating Deployer instance: {DeployerClass.__name__} for provider {provider_key}")
        
        try:
            # Hardening 4: Khởi tạo với config (sẽ được adapter xử lý để lấy credentials/params)
            return DeployerClass(**config)
        except Exception as e:
            logger.critical(f"Failed to instantiate deployer '{platform_type}' with config {config}: {e}")
            raise RuntimeError(f"Deployment Factory failed to create deployer: {e}")