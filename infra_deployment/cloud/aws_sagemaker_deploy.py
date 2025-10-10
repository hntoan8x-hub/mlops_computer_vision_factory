# cv_factory/infra/cloud/aws_sagemaker_deploy.py

import logging
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class AWSSageMakerDeployer:
    """
    A Façade class for deploying models to AWS SageMaker endpoints.

    This class handles the creation of the SageMaker Model, Endpoint Configuration,
    and the final Endpoint, ensuring the process is auditable and scalable.
    """

    def __init__(self, region_name: str, **kwargs: Dict[str, Any]):
        """
        Initializes the AWS client session.
        """
        self.region = region_name
        self.session = boto3.Session(region_name=self.region)
        self.sagemaker_client = self.session.client('sagemaker')
        logger.info(f"SageMaker client initialized for region {self.region}.")

    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Deploys a model (referenced by its artifact URI) to a SageMaker real-time endpoint.

        Args:
            model_name (str): The name for the SageMaker Model and Endpoint.
            model_artifact_uri (str): The S3 URI where the model artifact (e.g., MLflow model) is stored.
            deploy_config (Dict[str, Any]): Configuration details (e.g., instance_type, instance_count, execution_role).

        Returns:
            str: The name of the created SageMaker Endpoint.
        """
        execution_role = deploy_config['execution_role']
        image_uri = deploy_config['image_uri'] # ECR URI of the inference Docker image (cv-inference:latest)
        instance_type = deploy_config.get('instance_type', 'ml.m5.large')
        instance_count = deploy_config.get('instance_count', 1)
        
        try:
            # 1. Create SageMaker Model (References the inference container and model artifact)
            self.sagemaker_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=execution_role,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_artifact_uri # URI to the MLflow model artifact
                }
            )
            logger.info(f"SageMaker Model '{model_name}' created.")

            # 2. Create Endpoint Configuration
            endpoint_config_name = f"{model_name}-config"
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count,
                    'ServerlessConfig': deploy_config.get('serverless_config') # Optional Serverless Config
                }]
            )
            logger.info(f"Endpoint Configuration '{endpoint_config_name}' created.")

            # 3. Create Endpoint
            self.sagemaker_client.create_endpoint(
                EndpointName=model_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Deployment started for Endpoint '{model_name}'.")

            return model_name
        except ClientError as e:
            logger.error(f"AWS SageMaker deployment error: {e}")
            raise

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Deletes a SageMaker endpoint."""
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"SageMaker Endpoint '{endpoint_name}' deletion initiated.")
        except ClientError as e:
            logger.warning(f"Failed to delete endpoint {endpoint_name}. It might not exist: {e}")