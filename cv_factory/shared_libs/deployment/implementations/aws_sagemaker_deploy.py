# shared_libs/deployment/implementations/aws_sagemaker_deploy.py (FULL HARDENING)

import logging
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import asyncio
import time

from shared_libs.deployment.contracts.base_deployer import BaseDeployer 
from shared_libs.utils.exceptions import DeploymentError 
from shared_libs.deployment.configs.deployment_schema import SageMakerConfig 
from shared_libs.infra.secret_manager import SecretManagerClient 

logger = logging.getLogger(__name__)

class AWSSageMakerDeployer(BaseDeployer):
    """
    Adapter Façade for deploying models to AWS SageMaker endpoints.
    Tích hợp Secret Management để lấy Execution Role.
    """

    def __init__(self, config: Dict[str, Any]): 
        super().__init__(config)
        
        # Hardening 1: Khởi tạo Config từ Dict đã được validate
        self.sm_config = SageMakerConfig(**config) 
        self.region = self.sm_config.region_name
        
        # Hardening 2: Lấy Credentials từ Secret Manager
        secret_client = SecretManagerClient()
        sagemaker_secrets = secret_client.get_secret("aws_sagemaker_credentials")
        self.execution_role = sagemaker_secrets.get('execution_role')
        
        if not self.execution_role:
            raise ValueError("Could not retrieve 'execution_role' from Secret Manager. CRITICAL SECURITY FAILURE.")

        try:
            # Khởi tạo boto3 session
            self.session = boto3.Session(region_name=self.region)
            self.sagemaker_client = self.session.client('sagemaker')
            logger.info(f"SageMaker client initialized for region {self.region} using Secret Manager role.")
        except Exception as e:
            logger.critical(f"Failed to initialize AWS session or SageMaker client: {e}")
            raise DeploymentError(f"Deployment Adapter initialization failed: {e}")

    # --- Hardening 4: Thêm cơ chế chờ Endpoint InService ---
    def _wait_for_endpoint_status(self, endpoint_name: str, expected_status: str, timeout: int = 3600) -> None:
        """Chờ Endpoint chuyển sang trạng thái mong muốn (ví dụ: InService)."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                current_status = response['EndpointStatus']
                
                if current_status == expected_status:
                    logger.info(f"SageMaker Endpoint '{endpoint_name}' is now in status: {expected_status}.")
                    return
                
                if current_status in ['Failed', 'RollbackFailed']:
                    raise DeploymentError(f"SageMaker Endpoint '{endpoint_name}' deployment FAILED with status: {current_status}. Reason: {response.get('FailureReason')}")

                logger.info(f"Endpoint status: {current_status}. Waiting for {expected_status}...")
                time.sleep(30) # Đợi 30 giây

            except ClientError as e:
                logger.error(f"Error describing endpoint: {e}")
                raise
            except Exception as e:
                # Xử lý các lỗi khác
                raise DeploymentError(f"Error while waiting for endpoint status: {e}")

        raise DeploymentError(f"SageMaker Endpoint '{endpoint_name}' failed to reach status {expected_status} within {timeout} seconds.")
    
    
    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Deploys a model to a SageMaker real-time endpoint (tạo mới).
        """
        image_uri = deploy_config.get('image_uri') 
        instance_type = self.sm_config.instance_type
        instance_count = self.sm_config.instance_count
        
        if not image_uri:
            raise DeploymentError("Deployment configuration missing required 'image_uri'.")
        
        # 1. Create SageMaker Model
        self.sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=self.execution_role, 
            PrimaryContainer={'Image': image_uri, 'ModelDataUrl': model_artifact_uri}
        )
        logger.info(f"SageMaker Model '{model_name}' created.")

        # 2. Create Endpoint Configuration
        endpoint_config_name = f"{model_name}-config-{self.region}"
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'StableVariant', # Hardening: Đặt tên variant rõ ràng
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': instance_count,
                'InitialVariantWeight': 1.0 # 100% traffic ban đầu
            }]
        )
        logger.info(f"Endpoint Configuration '{endpoint_config_name}' created.")

        # 3. Create Endpoint
        self.sagemaker_client.create_endpoint(
            EndpointName=model_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Deployment started for Endpoint '{model_name}'.")

        # --- Quality Gate: Chờ Endpoint Ready ---
        self._wait_for_endpoint_status(model_name, 'InService')
        
        return model_name

    async def async_update_endpoint(self, endpoint_name: str, new_version_tag: str, deploy_config: Dict[str, Any]) -> None:
        """
        Cập nhật Endpoint hiện có để thêm Production Variant mới (Canary/Blue-Green).
        Sử dụng tag version làm tên Model/Variant mới.
        """
        new_model_name = f"{endpoint_name}-{new_version_tag}"
        new_variant_name = f"CanaryVariant-{new_version_tag}"
        image_uri = deploy_config.get('image_uri')
        instance_type = self.sm_config.instance_type
        instance_count = self.sm_config.instance_count
        
        # 1. Create NEW Model
        self.sagemaker_client.create_model(
            ModelName=new_model_name,
            ExecutionRoleArn=self.execution_role,
            PrimaryContainer={'Image': image_uri, 'ModelDataUrl': deploy_config['model_artifact_uri']}
        )
        logger.info(f"New SageMaker Model '{new_model_name}' created.")

        # 2. Update Endpoint (Adds the new variant with 0 weight)
        # Sử dụng UpdateEndpoint để thêm variant mới
        self.sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            DeploymentConfig={
                'BlueGreenUpdatePolicy': {
                    'TrafficRoutingConfiguration': {
                        'Type': 'ALL_AT_ONCE' # Sẽ dùng TrafficController (Service Mesh) để điều khiển, không dùng tính năng built-in của SM
                    },
                    'TerminationWaitInSeconds': 0 
                },
                'AutoRollbackConfiguration': {
                    'Alarms': [] # Tùy chọn: Thêm CloudWatch Alarms cho Auto Rollback
                }
            },
            # Hardening: Thêm cấu hình variant mới vào Endpoint Config
            EndpointConfigName=f"{endpoint_name}-config-{self.region}" # Tạo Config mới nếu cần
        )
        
        logger.info(f"SageMaker Endpoint update initiated (adding variant {new_variant_name}).")
        
        # Chờ Endpoint hoàn tất việc cập nhật
        self._wait_for_endpoint_status(endpoint_name, 'InService')


    def rollback(self, endpoint_name: str, target_version: str) -> None:
        """
        Rollback SageMaker bằng cách chuyển 100% traffic về Variant ổn định (StableVariant).
        """
        # Hardening: Rollback chỉ cần set lại trọng số (weight) cho Variant ổn định
        stable_variant_name = 'StableVariant'
        
        try:
            logger.info(f"Initiating Rollback for {endpoint_name}: Setting 100% traffic to {stable_variant_name}.")
            
            # Cập nhật trọng số của các variant: Stable = 1.0, Canary = 0.0
            self.sagemaker_client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[
                    {'VariantName': stable_variant_name, 'DesiredWeight': 1.0}
                    # Variants khác (Canary) sẽ tự động về 0 nếu không được đề cập
                ]
            )
            
            # Chờ Endpoint hoàn tất việc cập nhật Rollback
            self._wait_for_endpoint_status(endpoint_name, 'InService')
            logger.info(f"Rollback SUCCESSFUL. Traffic is 100% on {stable_variant_name}.")
            
        except ClientError as e:
            logger.critical(f"FATAL ROLLBACK FAILURE: Failed to update endpoint weights for {endpoint_name}: {e}")
            raise DeploymentError(f"SageMaker Rollback failed: {e}") from e

    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Xóa một Endpoint SageMaker, bao gồm Endpoint Configuration và Model liên kết.
        (CRITICAL COST MANAGEMENT: Đảm bảo xóa tất cả các thành phần)
        """
        try:
            # 1. Xóa Endpoint (Đây là tài nguyên tốn kém nhất)
            logger.info(f"Initiating deletion of SageMaker Endpoint: '{endpoint_name}'.")
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # 2. Xóa Endpoint Configuration
            # SageMaker thường yêu cầu tên cấu hình phải khớp với tên được sử dụng khi tạo
            endpoint_config_name = f"{endpoint_name}-config-{self.region}"
            logger.info(f"Deleting associated Endpoint Configuration: '{endpoint_config_name}'.")
            self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            
            # 3. Xóa SageMaker Model
            logger.info(f"Deleting associated SageMaker Model: '{endpoint_name}'.")
            self.sagemaker_client.delete_model(ModelName=endpoint_name)

            logger.info(f"SageMaker endpoint cleanup successful for '{endpoint_name}'.")

        except ClientError as e:
            # Hardening: Bắt lỗi ClientError (Ví dụ: Resource not found)
            if 'ResourceNotFound' in str(e):
                logger.warning(f"SageMaker resource for '{endpoint_name}' not found. Already deleted or never existed.")
            else:
                logger.error(f"Failed to delete SageMaker resources for '{endpoint_name}': {e}")
                # Vẫn raise lỗi nếu là lỗi nghiêm trọng khác
                raise DeploymentError(f"SageMaker deletion failed: {e}") from e