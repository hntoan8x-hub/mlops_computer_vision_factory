# cv_factory/infra/cloud/azure_ml_deploy.py

import logging
from typing import Dict, Any
# NOTE: Using azure-ai-ml SDK v2 (recommended for production)
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, Environment, Deployment, Endpoint
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

class AzureMLDeployer:
    """
    A Façade class for deploying models to Azure Machine Learning (AML) Endpoints.

    Handles model registration, environment (container) creation, and endpoint deployment.
    """
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initializes the Azure ML Client using default credentials.
        """
        try:
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            logger.info("Azure ML client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML client: {e}")
            raise

    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Deploys a model to an Azure Managed Online Endpoint.

        Args:
            model_name (str): The name for the AML Model and Endpoint.
            model_artifact_uri (str): The path/URI to the model artifact (e.g., AML Registered Model Asset ID).
            deploy_config (Dict[str, Any]): Configuration details (e.g., instance_type, compute_instance, environment_image).

        Returns:
            str: The name of the created Online Endpoint.
        """
        try:
            # 1. Register Model (Required to deploy)
            model_asset = Model(
                name=model_name,
                path=model_artifact_uri,
                type="mlflow_model" if model_artifact_uri.endswith(".zip") else "custom"
            )
            ml_model = self.ml_client.models.create_or_update(model_asset)
            logger.info(f"AML Model '{ml_model.name}' registered.")

            # 2. Create Environment (Refers to the inference Docker image)
            environment_name = deploy_config.get('environment_name', f"{model_name}-env")
            env_asset = Environment(
                name=environment_name,
                image=deploy_config['image_uri'] # ACR URI of the inference Docker image
            )
            ml_environment = self.ml_client.environments.create_or_update(env_asset)
            logger.info(f"AML Environment '{ml_environment.name}' ready.")

            # 3. Create Online Endpoint
            endpoint = Endpoint(name=model_name, auth_mode="key")
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
            logger.info(f"AML Endpoint '{model_name}' created.")

            # 4. Create Online Deployment
            deployment = Deployment(
                name="blue", # Standard deployment slot name
                endpoint_name=model_name,
                model=ml_model,
                environment=ml_environment,
                instance_type=deploy_config.get('instance_type', 'Standard_DS3_v2'),
                instance_count=deploy_config.get('instance_count', 1)
            )
            self.ml_client.online_deployments.begin_create_or_update(deployment).wait()
            self.ml_client.online_endpoints.begin_invoke(endpoint_name=model_name, deployment_name="blue")
            
            logger.info(f"Deployment to endpoint '{model_name}' complete.")
            return model_name
        except Exception as e:
            logger.error(f"Azure ML deployment error: {e}")
            raise

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Deletes an Azure ML online endpoint."""
        try:
            self.ml_client.online_endpoints.begin_delete(name=endpoint_name).wait()
            logger.info(f"AML Endpoint '{endpoint_name}' deletion initiated.")
        except Exception as e:
            logger.warning(f"Failed to delete endpoint {endpoint_name}. It might not exist: {e}")