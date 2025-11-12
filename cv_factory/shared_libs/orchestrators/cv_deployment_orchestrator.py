# shared_libs/orchestrators/cv_deployment_orchestrator.py (HARDENED)

import logging
import asyncio
from typing import Dict, Any, Union, List, Literal, Optional
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 

# Import Base Abstraction và Exceptions
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError

# Imports CONTRACTS (Chúng ta chỉ quan tâm đến Contracts, không phải Factory)
from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.deployment.contracts.base_traffic_controller import BaseTrafficController # Contract Traffic Controller

# LOẠI BỎ: from shared_libs.deployment.factory.deployer_factory import DeployerFactory
# LOẠI BỎ: from shared_libs.deployment.factory.traffic_controller_factory import TrafficControllerFactory
from shared_libs.ml_core.configs.orchestrator_config_schema import DeploymentOrchestratorConfig # Giả định Schema

logger = logging.getLogger(__name__)

class CVDeploymentOrchestrator(BaseOrchestrator):
    """
    Orchestrates the model deployment pipeline, supporting standard, canary, and rollback workflows.
    
    HARDENED: This class now receives its Deployer and Traffic Controller as injected dependencies.
    """

    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 logger_service: BaseTracker, 
                 event_emitter: Any,
                 deployer: BaseDeployer, # <<< INJECTED DEPLOYER >>>
                 traffic_controller: Optional[BaseTrafficController] = None # <<< INJECTED TRAFFIC CONTROLLER >>>
                 ):
        
        # 1. Base Init (Validation, Logging, Event Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        # 2. Store Injected Dependencies
        self.deployer: BaseDeployer = deployer
        self.traffic_controller: Optional[BaseTrafficController] = traffic_controller
        
        # 3. LOẠI BỎ logic khởi tạo nội bộ:
        # Ví dụ: self.deployer = DeployerFactory.create(...)
        # Ví dụ: self.traffic_controller = TrafficControllerFactory.create(...)
        
        self.logger.info(f"[{self.orchestrator_id}] Deployment Orchestrator initialized with Deployer: {type(deployer).__name__}")
        if self.traffic_controller:
             self.logger.info(f"[{self.orchestrator_id}] Traffic Controller: {type(traffic_controller).__name__} enabled.")


    # --- Mandatory Abstract Method Implementation ---

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validates the configuration using the Pydantic schema (Enforces Quality Gate).
        """
        try:
            # Assume config schema is DeploymentOrchestratorConfig
            DeploymentOrchestratorConfig(**config.get('deployment', {}))
            self.logger.info("Configuration validated successfully against DeploymentOrchestratorConfig schema.")
        except Exception as e:
            self.logger.error(f"Invalid deployment configuration detected: {e}")
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Deployment config validation failed: {e}") from e

    # --- Run Workflow Methods (Logic giữ nguyên, chỉ thay đổi nguồn dependency) ---

    def run(self, 
            model_artifact_uri: str, 
            model_name: str, 
            mode: Literal['standard', 'canary'] = 'standard', 
            new_version_tag: str = 'latest', 
            stable_version: Optional[str] = None
            ) -> str:
        """
        Executes the deployment workflow (Standard or Canary).
        """
        self.logger.info(f"[{self.orchestrator_id}] Starting deployment workflow in mode: {mode}")

        if mode == 'standard':
            endpoint_id = asyncio.run(self._standard_deployment(
                model_artifact_uri, model_name, new_version_tag, stable_version
            ))
        elif mode == 'canary':
            endpoint_id = asyncio.run(self._canary_rollout(
                model_artifact_uri, model_name, new_version_tag, stable_version
            ))
        else:
            raise InvalidConfigError(f"Unsupported deployment mode: {mode}")

        self.log_metrics({"deployment_mode": mode, "model_version": new_version_tag})
        return endpoint_id

    # ... Các phương thức _standard_deployment và _canary_rollout (giả định)
    # ... Các phương thức này vẫn sử dụng self.deployer và self.traffic_controller

    async def _standard_deployment(self, model_uri: str, model_name: str, new_version: str, stable_version: Optional[str]) -> str:
        """Performs a direct deployment (No traffic shifting)."""
        deploy_config = self.config.get('deployment', {}).get('deploy_params', {})
        endpoint_name = self.config.get('deployment', {}).get('endpoint_name', model_name)

        self.logger.info(f"Standard deployment of {model_name} v{new_version} to endpoint {endpoint_name}.")

        # 1. Triển khai phiên bản mới
        await self.deployer.async_update_endpoint(
            endpoint_name=endpoint_name, 
            model_uri=model_uri, 
            new_version_tag=new_version, 
            deploy_config=deploy_config
        )

        # 2. Switch Traffic (Nếu có Controller)
        if self.traffic_controller:
            self.logger.info("Switching 100% traffic to new version via Traffic Controller.")
            await self.traffic_controller.async_set_traffic(
                new_version=new_version, 
                new_traffic_percentage=100, 
                stable_version=stable_version or new_version
            )
        
        self.logger.info(f"Deployment successful. Endpoint: {endpoint_name}")
        return endpoint_name


    async def _canary_rollout(self, model_uri: str, model_name: str, new_version: str, stable_version: str) -> str:
        """Performs a Canary deployment (Traffic Shifting)."""
        if not self.traffic_controller:
            raise WorkflowExecutionError("Canary deployment requires a configured Traffic Controller.")

        # Logic cũ từ snippet (đã được làm sạch để sử dụng self.deployer/self.traffic_controller)
        deploy_config = self.config.get('deployment', {}).get('deploy_params', {})
        endpoint_name = self.config.get('deployment', {}).get('endpoint_name', model_name)
        canary_traffic = self.config.get('deployment', {}).get('canary_traffic_percent', 10)

        self.logger.info(f"Starting Canary rollout for {model_name} v{new_version}. Initial traffic: {canary_traffic}%")
        
        # 1. Triển khai phiên bản mới (Deployment/Endpoint Update)
        await self.deployer.async_update_endpoint(
            endpoint_name=endpoint_name,
            model_uri=model_uri,
            new_version_tag=new_version,
            deploy_config=deploy_config
        )
        
        # 2. Chuyển Lưu lượng (Traffic Switch)
        self.logger.info(f"Switching {canary_traffic}% traffic via {self.traffic_controller.__class__.__name__}.")
        
        success = await self.traffic_controller.async_set_traffic(
            new_version=new_version, 
            new_traffic_percentage=canary_traffic, 
            stable_version=stable_version
        )
        
        if not success:
            self.logger.critical("Canary traffic switch failed. Initiating automatic rollback.")
            self.deployer.rollback(endpoint_name, stable_version)
            raise WorkflowExecutionError("Canary deployment failed during traffic switch.")
        
        self.logger.info(f"Canary deployed successfully at {canary_traffic}%. Waiting for QA feedback before full rollout.")
        return endpoint_name