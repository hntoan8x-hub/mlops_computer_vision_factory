# cv_factory/api_service/clients/cloud_inference_client.py

import logging
import requests
import base64
from typing import Dict, Any, Union, Optional

# Import the Configuration Store
from cv_factory.api_service.api_config import API_CONFIG 

# Import Exception
from shared_libs.core_utils.exceptions import WorkflowExecutionError
# Import Deployers (optional utility, kept for completeness)
from cv_factory.infra.cloud.aws_sagemaker_deploy import AWSSageMakerDeployer 
from cv_factory.infra.cloud.gcp_vertex_deploy import GCPVertexDeployer 
from cv_factory.infra.cloud.azure_ml_deploy import AzureMLDeployer 

logger = logging.getLogger(__name__)

class CloudInferenceClient:
    """
    Adapter client for invoking a deployed model endpoint on a managed cloud service.

    This client abstracts the specific API calls and payload formats, fetching all
    operational parameters (URL, provider, token) from the central API_CONFIG.
    """
    
    def __init__(self):
        """
        Initializes the client by loading configuration from API_CONFIG.
        
        No parameters are required in __init__ because all critical operational 
        settings are sourced from the global API_CONFIG object (Singleton pattern).
        """
        self.endpoint_url = API_CONFIG.endpoint_url
        self.provider = API_CONFIG.cloud_provider.lower()
        self.region = API_CONFIG.region # Assuming region is also part of API_CONFIG
        self.auth_token = API_CONFIG.auth_token

        # Set up HTTP session
        self.session = requests.Session()
        self.session.headers.update(self._get_auth_headers())
        
        logger.info(f"Cloud Inference Client initialized. Target: {self.provider} at {self.endpoint_url}")

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Generates authentication headers specific to the cloud provider using the stored token.
        """
        headers = {'Content-Type': 'application/json'}
        
        if self.provider == 'aws':
             # AWS signature V4 logic is complex, often using a dedicated requests library, 
             # but here we ensure the content type is correct.
             pass
        
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
            
        return headers

    def invoke_endpoint(self, payload: bytes, threshold: float) -> Dict[str, Any]:
        """
        Invokes the deployed model endpoint with the given payload (image bytes).
        """
        if not self.endpoint_url:
            raise WorkflowExecutionError("Endpoint URL is missing. Check API configuration.")
            
        # 1. Prepare JSON Payload (standardized for cloud endpoints)
        payload_base64 = base64.b64encode(payload).decode('utf-8')
        
        request_body = {
            "instances": [{"image_data": payload_base64}],
            "parameters": {"threshold": threshold}
        }
        
        try:
            # 2. Send Request to Cloud Endpoint
            response = self.session.post(self.endpoint_url, json=request_body)
            response.raise_for_status() 
            
            # 3. Process Response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Cloud endpoint invocation failed for {self.provider}: {e}")
            raise WorkflowExecutionError(f"Cloud Endpoint Invocation Error: {e}")
        except Exception as e:
            logger.error(f"Error processing endpoint response: {e}")
            raise WorkflowExecutionError(f"API Response Processing Error: {e}")

    # --- Utility for Deployment (Optional - Kept for End-to-End completeness) ---

    @classmethod
    def deploy_new_model(cls, cloud_provider: str, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Utility method to call the appropriate deployer from infra/cloud/.
        """
        # This method would typically use API_CONFIG for region/project/subscription details
        # when initializing the specific deployer.
        if cloud_provider.lower() == 'aws':
            deployer = AWSSageMakerDeployer(region_name=API_CONFIG.region)

    # --- Utility for Deployment (Optional - Can be used by CI/CD or specialized routes) ---

    @classmethod
    def deploy_new_model(cls, cloud_provider: str, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Utility method to call the appropriate deployer from infra/cloud/.
        """
        if cloud_provider.lower() == 'aws':
            deployer = AWSSageMakerDeployer(region_name=deploy_config['region'])
        elif cloud_provider.lower() == 'gcp':
            deployer = GCPVertexDeployer(project=deploy_config['project_id'], location=deploy_config['location'])
        elif cloud_provider.lower() == 'azure':
            deployer = AzureMLDeployer(subscription_id=deploy_config['subscription_id'], resource_group=deploy_config['resource_group'], workspace_name=deploy_config['workspace_name'])
        else:
            raise ValueError(f"Unsupported cloud provider for deployment: {cloud_provider}")
            
        return deployer.deploy_model(model_name, model_artifact_uri, deploy_config)