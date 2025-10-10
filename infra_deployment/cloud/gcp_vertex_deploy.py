# cv_factory/infra/cloud/gcp_vertex_deploy.py

import logging
from typing import Dict, Any
# NOTE: Using Google Cloud Vertex AI SDK
from google.cloud import aiplatform
from google.protobuf.json_format import ParseDict

logger = logging.getLogger(__name__)

class GCPVertexDeployer:
    """
    A Façade class for deploying models to Google Cloud Vertex AI Endpoints.

    Handles model upload, endpoint creation, and deployment.
    """
    def __init__(self, project: str, location: str):
        """
        Initializes the Vertex AI client.
        """
        self.project = project
        self.location = location
        aiplatform.init(project=self.project, location=self.location)
        logger.info(f"Vertex AI client initialized for project {self.project} in {self.location}.")

    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Deploys a model to a Vertex AI Endpoint.

        Args:
            model_name (str): The name for the Vertex AI Model and Endpoint.
            model_artifact_uri (str): The GCS URI to the model artifact (e.g., MLflow model path).
            deploy_config (Dict[str, Any]): Configuration details (e.g., machine_type, container_uri).

        Returns:
            str: The name of the deployed Endpoint resource.
        """
        container_uri = deploy_config['image_uri'] # GCR/Artifact Registry URI of the inference Docker image
        machine_type = deploy_config.get('machine_type', 'n1-standard-4')
        min_replica_count = deploy_config.get('min_replica_count', 1)
        max_replica_count = deploy_config.get('max_replica_count', 1)
        
        try:
            # 1. Upload Model to Vertex AI Model Registry
            uploaded_model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=model_artifact_uri,
                serving_container_image_uri=container_uri,
                description=f"CV Model deployed from MLOps Factory run."
            )
            logger.info(f"Vertex AI Model '{uploaded_model.name}' uploaded.")

            # 2. Create Endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=f"{model_name}-endpoint",
                project=self.project,
                location=self.location
            )
            logger.info(f"Vertex AI Endpoint '{endpoint.name}' created.")

            # 3. Deploy Model to Endpoint
            traffic_split = deploy_config.get('traffic_split', {"0": 100})
            
            uploaded_model.deploy(
                endpoint=endpoint,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                traffic_split=traffic_split
            )
            logger.info(f"Deployment started on Endpoint '{endpoint.name}'.")

            return endpoint.name
        except Exception as e:
            logger.error(f"GCP Vertex AI deployment error: {e}")
            raise

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Deletes a Vertex AI endpoint."""
        try:
            # Find the endpoint by display name (or resource ID) and undeploy all
            endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
            endpoint.undeploy_all()
            endpoint.delete()
            logger.info(f"Vertex AI Endpoint '{endpoint_name}' deletion initiated.")
        except Exception as e:
            logger.warning(f"Failed to delete endpoint {endpoint_name}. It might not exist: {e}")