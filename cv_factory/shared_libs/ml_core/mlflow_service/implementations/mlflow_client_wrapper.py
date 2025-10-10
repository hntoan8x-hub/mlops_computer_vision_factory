import logging
from typing import Optional
import mlflow
from shared_libs.ml_core.mlflow_service.utils.retry_utils import retry

logger = logging.getLogger(__name__)

class MLflowClientWrapper:
    """
    A wrapper class for the MLflow client to handle connection and configuration.

    It ensures a single, reusable client instance throughout the application.
    """
    _instance = None
    _client_instance = None

    def __new__(cls, tracking_uri: Optional[str] = None):
        """Singleton pattern to ensure only one client wrapper instance exists."""
        if cls._instance is None:
            cls._instance = super(MLflowClientWrapper, cls).__new__(cls)
            cls._instance.tracking_uri = tracking_uri
            cls._instance._init_client()
        return cls._instance

    @retry(retries=3)
    def _init_client(self) -> None:
        """Initializes the MLflow client and sets the tracking URI."""
        if self._client_instance is None:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            self._client_instance = mlflow.tracking.MlflowClient()
            logger.info(f"MLflow client initialized with URI: {mlflow.get_tracking_uri()}")

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        """Returns the MLflow client instance."""
        if self._client_instance is None:
            self._init_client()
        return self._client_instance