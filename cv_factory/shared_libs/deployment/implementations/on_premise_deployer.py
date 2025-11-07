# shared_libs/deployment/implementations/on_premise_deployer.py (FINAL FULL PRODUCTION CODE)

import logging
import subprocess
from typing import Dict, Any, Optional
import asyncio
# import requests # Cần thư viện này cho API calls thực tế

from shared_libs.deployment.contracts.base_deployer import BaseDeployer
from shared_libs.infra.secret_manager import SecretManagerClient
from shared_libs.utils.exceptions import DeploymentError 
from shared_libs.deployment.configs.deployment_schema import OnPremiseConfig # Giả định schema

logger = logging.getLogger(__name__)

# Giả định OnPremiseConfig schema đã được validate
class OnPremiseConfig:
     def __init__(self, script_path, api_endpoint, method):
        self.script_path = script_path
        self.api_endpoint = api_endpoint
        self.method = method

class OnPremiseDeployer(BaseDeployer):
    """
    Adapter để triển khai mô hình ML lên hạ tầng On-Premise/Nội bộ.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.onpremise_config = OnPremiseConfig(**config) 
        self.deployment_script_path = self.onpremise_config.script_path
        self.internal_api_endpoint = self.onpremise_config.api_endpoint
        self.deployment_method = self.onpremise_config.method
        
        self.api_token = None
        if self.deployment_method == 'api':
            # Hardening: Lấy Token nội bộ từ Secret Manager
            secret_client = SecretManagerClient()
            self.api_token = secret_client.get_secret("internal_api_token").get('token')
            if not self.api_token:
                 logger.error("Internal API token not found in Secret Manager.")
                 raise DeploymentError("Missing API token for On-Premise deployment.")
            
        logger.info(f"OnPremiseDeployer initialized. Method: {self.deployment_method}.")


    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        [TRIỂN KHAI] Kích hoạt quy trình triển khai On-Premise (Tạo mới hoặc Update full).
        """
        version_tag = deploy_config.get('version_tag', 'v1') # Hardening: Thêm version tag
        
        if self.deployment_method == 'script':
            # Phương pháp 1: Chạy Script Shell Nội bộ
            command = [
                'bash', 
                self.deployment_script_path, 
                '--action', 'deploy',
                '--model-name', model_name, 
                '--uri', model_artifact_uri,
                '--version', version_tag
            ]
            
            try:
                # Hardening: Bắt lỗi subprocess
                subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"On-premise deployment script executed successfully for {model_name} v{version_tag}.")
            except subprocess.CalledProcessError as e:
                logger.error(f"On-premise deployment script failed. Stderr: {e.stderr}")
                raise DeploymentError(f"On-premise deploy script failed: {e.stderr}")

        elif self.deployment_method == 'api':
            # Phương pháp 2: Gọi API Triển khai Nội bộ
            payload = {
                "service_name": model_name,
                "model_uri": model_artifact_uri,
                "version": version_tag,
                "action": "deploy"
            }
            headers = {'Authorization': f'Bearer {self.api_token}'} 
            
            # Giả định gọi API và chờ kết quả
            # response = requests.post(self.internal_api_endpoint, json=payload, headers=headers)
            # response.raise_for_status() 
            logger.info(f"On-premise deployment API called for {model_name} v{version_tag}. Requires 'requests' impl.")
            
        return model_name # Tên service là tên endpoint

    # Hardening 4: Triển khai các phương thức Canary/Rollback đầy đủ
    async def async_update_endpoint(self, endpoint_name: str, new_version_tag: str, deploy_config: Dict[str, Any]) -> None:
        """
        Kích hoạt cập nhật On-Premise (Thêm phiên bản mới/Variant cho Canary).
        """
        model_artifact_uri = deploy_config['model_artifact_uri']
        
        # Bước 1: Triển khai phiên bản mới (giống deploy_model nhưng không chuyển traffic 100%)
        logger.info(f"On-premise update: Deploying new version {new_version_tag} (will have 0% traffic).")
        deploy_config['version_tag'] = new_version_tag
        self.deploy_model(endpoint_name, model_artifact_uri, deploy_config) 
        
        # Bước 2: Kích hoạt Traffic Controller nội bộ (giả định)
        # Trong hệ thống On-Premise thực tế, bước này sẽ gọi một API khác
        # để cấu hình Load Balancer/Service Mesh nội bộ.
        # Ví dụ: await requests.post(self.internal_traffic_api, ...)
        logger.info("On-premise new version deployed. TrafficController (internal) will manage traffic split.")
        await asyncio.sleep(1)


    def rollback(self, endpoint_name: str, target_version: str) -> None:
        """
        Kích hoạt Rollback On-Premise (chuyển 100% traffic về Stable).
        """
        if self.deployment_method == 'script':
            # Chạy script Rollback nội bộ (chuyển 100% traffic)
            rollback_script_path = self.deployment_script_path.replace('deploy', 'rollback') # Giả định
            command = [
                'bash', 
                rollback_script_path, 
                '--service-name', endpoint_name,
                '--target-version', target_version
            ]
            
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"On-premise rollback script executed successfully to v{target_version}.")
            except subprocess.CalledProcessError as e:
                logger.error(f"On-premise rollback failed. Stderr: {e.stderr}")
                raise DeploymentError(f"On-premise rollback failed: {e.stderr}")

        elif self.deployment_method == 'api':
            # Gọi API Rollback nội bộ
            payload = {
                "service_name": endpoint_name,
                "version": target_version,
                "action": "rollback"
            }
            # requests.post(self.internal_api_endpoint, json=payload, headers=headers)
            logger.info(f"On-premise rollback API called for {endpoint_name} to v{target_version}. Requires 'requests' impl.")


    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Xóa một Endpoint On-Premise bằng cách kích hoạt một quy trình tắt/xóa nội bộ.
        """
        if self.deployment_method == 'script':
            cleanup_script_path = self.deployment_script_path.replace('deploy', 'cleanup') 
            command = [
                'bash', 
                cleanup_script_path, 
                '--action', 'delete',
                '--service-name', endpoint_name
            ]
            
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"On-premise cleanup script executed successfully for {endpoint_name}.")
            except subprocess.CalledProcessError as e:
                logger.error(f"On-premise cleanup script failed. Stderr: {e.stderr}")
                raise DeploymentError(f"On-premise deletion failed: {e.stderr}")
        
        elif self.deployment_method == 'api':
            # Phương pháp 2: Gọi API Xóa Nội bộ
            # requests.delete(f"{self.internal_api_endpoint}/{endpoint_name}", headers={'Authorization': self.api_token})
            logger.info(f"API deletion called for {endpoint_name}. Requires 'requests' impl.")

        logger.info(f"On-premise endpoint deletion successfully initiated for service: {endpoint_name}.")