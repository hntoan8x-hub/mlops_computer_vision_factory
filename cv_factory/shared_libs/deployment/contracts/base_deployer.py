# shared_libs/deployment/contracts/base_deployer.py
# -*- coding: utf-8 -*-

import abc
from typing import Dict, Any, Optional

class BaseDeployer(abc.ABC):
    """
    Abstract Base Class (Contract) for all Model Deployment Services (Cloud or On-Premise).

    Defines a standard interface for deploying and managing real-time inference endpoints.
    This contract enforces Cloud-Agnostic principles.
    """

    @abc.abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initializes the deployer client with region/project/workspace details."""
        self.config = config
        pass

    @abc.abstractmethod
    def deploy_model(self, model_name: str, model_artifact_uri: str, deploy_config: Dict[str, Any]) -> str:
        """
        Deploys a model artifact to a real-time endpoint.

        Args:
            model_name (str): The unique name for the endpoint/resource.
            model_artifact_uri (str): URI to the model artifact (S3, GCS, MLflow).
            deploy_config (Dict[str, Any]): Configuration specific to the deployment environment 
                                            (e.g., instance_type, container_image_uri).

        Returns:
            str: The name or unique ID of the created endpoint.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete_endpoint(self, endpoint_name: str) -> None:
        """Deletes a deployed endpoint and associated resources."""
        raise NotImplementedError