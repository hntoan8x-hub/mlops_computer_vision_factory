# shared_libs/orchestrators/cv_deployment_orchestrator.py

import logging
import asyncio
from typing import Dict, Any, Union, List, Literal, Optional

# Import Base Abstraction và Exceptions
from shared_libs.orchestrators.base.base_orchestrator import BaseOrchestrator
from shared_libs.orchestrators.utils.orchestrator_exceptions import InvalidConfigError, WorkflowExecutionError
from shared_libs.deployment.factory.deployer_factory import DeployerFactory
from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.deployment.contracts.base_traffic_controller import BaseTrafficController # Contract Traffic Controller
from shared_libs.deployment.factory.traffic_controller_factory import TrafficControllerFactory # Giả định Factory này tồn tại
from shared_libs.ml_core.configs.orchestrator_config_schema import DeploymentOrchestratorConfig # Giả định Schema

logger = logging.getLogger(__name__)

class CVDeploymentOrchestrator(BaseOrchestrator):
    """
    Orchestrates the model deployment pipeline, supporting standard, canary, and rollback workflows.
    
    This class utilizes the DeployerFactory and coordinates with the TrafficController
    to ensure zero-downtime model updates.
    """

    def __init__(self, orchestrator_id: str, config: Dict[str, Any], logger_service: Any, event_emitter: Any):
        
        # 1. Base Init (Validation, Logging, Event Emitter Injection)
        super().__init__(orchestrator_id, config, logger_service, event_emitter)
        
        # 2. Khởi tạo Deployer
        platform = self.config['deployment']['platform']
        deployer_config = self.config['deployment']['config']
        
        self.deployer: BaseDeployer = DeployerFactory.create_deployer(
            platform_type=platform, 
            config=deployer_config
        )
        
        # 3. Khởi tạo Traffic Controller (Nếu cần)
        self.traffic_controller: Optional[BaseTrafficController] = self._initialize_traffic_controller(platform, deployer_config)
        
        self.logger.info(f"Deployment Orchestrator initialized. Platform: {platform}.")


    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validates the configuration using the Pydantic schema."""
        try:
            DeploymentOrchestratorConfig(**config) 
            self.logger.info("Configuration validated successfully against DeploymentOrchestratorConfig.")
        except Exception as e:
            self.emit_event(event_name="config_validation_failure", payload={"error": str(e)})
            raise InvalidConfigError(f"Deployment config validation failed: {e}") from e


    def _initialize_traffic_controller(self, platform: str, config: Dict[str, Any]) -> Optional[BaseTrafficController]:
        """
        Khởi tạo Traffic Controller chỉ khi nền tảng hỗ trợ (ví dụ: Kubernetes/Istio).
        """
        traffic_config = config.get('traffic_controller', {})
        if traffic_config:
            try:
                # Giả định TrafficControllerFactory tồn tại và được cấu hình
                return TrafficControllerFactory.create_controller(
                    platform_type=traffic_config.get('type', 'istio'), 
                    endpoint_name=self.config['deployment']['endpoint_name'],
                    config=traffic_config
                )
            except Exception as e:
                 # Traffic Controller không bắt buộc, nhưng nếu config, phải khởi tạo thành công
                 self.logger.warning(f"Could not initialize Traffic Controller: {e}. Running without canary support.")
                 return None
        return None

    # --- Core Execution Flow ---

    def run(self, 
            model_artifact_uri: str, 
            model_name: str, 
            mode: Literal["standard", "canary", "rollback"] = "standard", 
            **kwargs: Dict[str, Any]
        ) -> str:
        """
        [IMPLEMENTED] Executes the deployment workflow based on the specified mode.
        """
        self.logger.info(f"[{self.orchestrator_id}] Starting deployment in mode: {mode}.")
        endpoint_name = self.config['deployment']['endpoint_name']
        deploy_config = self.config['deployment']['config']
        deploy_config['model_artifact_uri'] = model_artifact_uri # Thêm URI cho Deployer
        
        try:
            if mode == "standard":
                # CHẾ ĐỘ 1: TRIỂN KHAI HOẶC CẬP NHẬT 100% TRAFFIC (zero-downtime)
                endpoint = self.deployer.deploy_model(model_name, model_artifact_uri, deploy_config)
                self.logger.info(f"Standard deployment completed. Endpoint: {endpoint}.")
                return endpoint
                
            elif mode == "canary" and self.traffic_controller:
                return asyncio.run(self._run_canary_deployment(model_name, endpoint_name, deploy_config, **kwargs))
            
            elif mode == "rollback":
                if not kwargs.get('target_version'):
                    raise ValueError("Rollback mode requires 'target_version'.")
                self.deployer.rollback(endpoint_name, kwargs['target_version'])
                return endpoint_name # Trả về tên Endpoint
            
            else:
                self.logger.warning(f"Mode '{mode}' not supported or Traffic Controller not configured. Falling back to standard deployment.")
                return self.deployer.deploy_model(model_name, model_artifact_uri, deploy_config)
            
        except Exception as e:
            error_msg = f"Deployment workflow failed: {type(e).__name__}."
            self.logger.error(error_msg, exc_info=True)
            self.emit_event(event_name="deployment_failure", payload={"error": error_msg})
            raise WorkflowExecutionError(error_msg) from e

    # --- Phương thức cho Canary Deployment ---

    async def _run_canary_deployment(self, model_name: str, endpoint_name: str, 
                                     deploy_config: Dict[str, Any], **kwargs: Dict[str, Any]) -> str:
        """
        Luồng Canary: Triển khai phiên bản mới, chuyển traffic từng bước.
        """
        new_version_tag = kwargs.get('new_version_tag', 'canary')
        canary_traffic = kwargs.get('canary_traffic_percent', 5) # 5% traffic ban đầu
        stable_version = kwargs.get('stable_version', 'stable')
        
        self.logger.info(f"Starting CANARY deployment: {stable_version} -> {new_version_tag} with {canary_traffic}% initial traffic.")
        
        # 1. Triển khai phiên bản mới (Deployment/Endpoint Update)
        # Sử dụng phương thức async_update_endpoint của Deployer (có trong SageMaker, K8s, On-Premise)
        await self.deployer.async_update_endpoint(endpoint_name, new_version_tag, deploy_config)
        
        # 2. Chuyển Lưu lượng (Traffic Switch)
        if self.traffic_controller:
            self.logger.info(f"Switching {canary_traffic}% traffic via {self.traffic_controller.__class__.__name__}.")
            
            # Cấu hình Traffic Controller: Chuyển một phần traffic sang phiên bản mới
            success = await self.traffic_controller.async_set_traffic(
                new_version=new_version_tag, 
                new_traffic_percentage=canary_traffic, 
                stable_version=stable_version
            )
            
            if not success:
                # Nếu thất bại, cần Rollback ngay lập tức (CRITICAL)
                self.logger.critical("Canary traffic switch failed. Initiating automatic rollback.")
                self.deployer.rollback(endpoint_name, stable_version)
                raise WorkflowExecutionError("Canary deployment failed during traffic switch.")
        
        # 3. Giữ nguyên (Pause) để kiểm tra chất lượng (Sau đó sẽ có luồng khác kích hoạt full rollout)
        self.logger.info(f"Canary deployed successfully at {canary_traffic}%. Waiting for QA feedback before full rollout.")
        
        return endpoint_name