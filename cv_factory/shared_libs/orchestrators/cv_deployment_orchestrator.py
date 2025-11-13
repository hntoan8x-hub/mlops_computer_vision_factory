# shared_libs/orchestrators/cv_deployment_orchestrator.py (UPDATED)

import logging
import asyncio
from typing import Dict, Any, Union, List, Literal, Optional
from shared_libs.ml_core.mlflow_service.base.base_tracker import BaseTracker 
from shared_libs.ml_core.mlflow_service.base.base_registry import BaseRegistry # NEW IMPORT

# Import Base Abstraction và Exceptions
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError

# Imports CONTRACTS
from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.deployment.contracts.base_traffic_controller import BaseTrafficController 
from shared_libs.ml_core.configs.orchestrator_config_schema import DeploymentOrchestratorConfig 

logger = logging.getLogger(__name__)

class CVDeploymentOrchestrator(BaseOrchestrator):
    """
    Orchestrates the model deployment pipeline, supporting standard, canary, and rollback workflows.
    """

    def __init__(self, 
                 orchestrator_id: str, 
                 config: Dict[str, Any], 
                 logger_service: BaseTracker, 
                 event_emitter: Any,
                 deployer: BaseDeployer, 
                 traffic_controller: Optional[BaseTrafficController] = None,
                 registry_service: Optional[BaseRegistry] = None # <<< INJECTED REGISTRY >>>
                 ):
        
        # 1. Base Init 
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        # 2. Store Injected Dependencies
        self.deployer: BaseDeployer = deployer
        self.traffic_controller: Optional[BaseTrafficController] = traffic_controller
        self.registry = registry_service # NEW: Store injected registry

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
    
    # --- NEW HELPER: Get Stable Version from Registry ---
    def _get_current_stable_version(self, model_name: str) -> Optional[str]:
        """
        [HARDENED] Retrieves the current stable version tag from the MLflow Registry.
        """
        if not self.registry:
             self.logger.warning("Registry service not available. Cannot check stable version.")
             return None
        
        try:
             # Giả định: Phiên bản ổn định được tag là 'Production'
             return self.registry.get_latest_version(model_name, stage='Production')
        except Exception as e:
             self.logger.error(f"Could not retrieve 'Production' version for {model_name}: {e}")
             return None

    # --- Run Workflow Methods ---

    def run(self, 
            model_artifact_uri: str, 
            model_name: str, 
            mode: Literal['standard', 'canary', 'rollback'] = 'standard', # NEW: Thêm 'rollback'
            new_version_tag: str = 'latest', 
            stable_version: Optional[str] = None # Stable version là version cũ (hoặc version đích cho rollback)
            ) -> str:
        """
        Executes the deployment workflow (Standard, Canary, or Rollback).
        """
        self.logger.info(f"[{self.orchestrator_id}] Starting deployment workflow in mode: {mode}")

        if mode == 'standard':
            # Nếu stable_version không được cung cấp, hãy cố gắng lấy phiên bản Production hiện tại
            stable_version = stable_version or self._get_current_stable_version(model_name)
            endpoint_id = asyncio.run(self._standard_deployment(
                model_artifact_uri, model_name, new_version_tag, stable_version
            ))
        elif mode == 'canary':
            # Canary luôn cần một stable_version để chuyển lưu lượng từ nó
            if not stable_version:
                 stable_version = self._get_current_stable_version(model_name)
            if not stable_version:
                 raise InvalidConfigError("Canary deployment requires a stable_version for traffic shifting.")
                 
            endpoint_id = asyncio.run(self._canary_rollout(
                model_artifact_uri, model_name, new_version_tag, stable_version
            ))
        elif mode == 'rollback':
             if not stable_version:
                 raise InvalidConfigError("Rollback requires a stable_version (target version) to roll back to.")
             endpoint_id = asyncio.run(self._rollback_deployment(
                 model_name, stable_version
             ))
        else:
            raise InvalidConfigError(f"Unsupported deployment mode: {mode}")

        self.log_metrics({"deployment_mode": mode, "model_version": new_version_tag})
        return endpoint_id

    # --- NEW METHOD: Rollback Deployment ---
    async def _rollback_deployment(self, model_name: str, target_version: str) -> str:
        """Performs a full traffic switch back to the specified target_version."""
        if not self.traffic_controller:
            raise WorkflowExecutionError("Rollback requires a configured Traffic Controller.")

        endpoint_name = self.config.get('deployment', {}).get('endpoint_name', model_name)
        self.logger.critical(f"Executing EMERGENCY ROLLBACK to version: {target_version}")

        # Deployer Rollback (Đảm bảo deployment của target version đang tồn tại)
        # Giả định deployer có phương thức rollback
        await self.deployer.async_rollback_to_version(endpoint_name, target_version) 

        # Switch Traffic 100% về phiên bản cũ
        await self.traffic_controller.async_set_traffic(
            new_version=target_version,
            new_traffic_percentage=100,
            stable_version=target_version # Target version trở thành stable version
        )
        
        # Cập nhật Registry Tag (Nếu có)
        if self.registry:
             self.registry.transition_model_version_stage(model_name, target_version, stage='Production')
             
        self.logger.info(f"Rollback successful. Endpoint: {endpoint_name}. Traffic is 100% on {target_version}.")
        return endpoint_name

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
                stable_version=stable_version or new_version # Sử dụng stable_version đã tìm được/hoặc chính nó
            )
        
        self.logger.info(f"Deployment successful. Endpoint: {endpoint_name}")
        return endpoint_name


    async def _canary_rollout(self, model_uri: str, model_name: str, new_version: str, stable_version: str) -> str:
        """Performs a Canary deployment (Traffic Shifting)."""
        if not self.traffic_controller:
            raise WorkflowExecutionError("Canary deployment requires a configured Traffic Controller.")

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
            # Rollback deployment
            await self.deployer.async_rollback_to_version(endpoint_name, stable_version)
            raise WorkflowExecutionError("Canary deployment failed during traffic switch.")
        
        self.logger.info(f"Canary deployed successfully at {canary_traffic}%. Waiting for QA feedback before full rollout.")
        return endpoint_name