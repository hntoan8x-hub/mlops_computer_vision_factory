# cv_factory/shared_libs/data_ingestion/connectors/api_connector.py

import logging
import requests
from typing import Dict, Any, Union, Optional
from ..base.base_data_connector import BaseDataConnector, OutputData

logger = logging.getLogger(__name__)

class APIConnector(BaseDataConnector):
    """
    Concrete connector for fetching data from and sending data to REST API endpoints.
    """

    def connect(self) -> bool:
        """
        Connect for API usually means verifying base URL accessibility or initializing 
        session headers (e.g., authentication tokens).
        """
        self.session = requests.Session()
        # Example: self.session.headers.update({'Authorization': f'Bearer {self.config.get("token")}'})
        self.is_connected = True
        logger.info(f"[{self.connector_id}] API Connector session established.")
        return True

    def read(self, source_uri: str, method: str = 'GET', data: Optional[Dict[str, Any]] = None, **kwargs) -> OutputData:
        """
        Fetches data from an API endpoint. Supports GET or POST methods for reading.
        """
        if not self.is_connected: self.connect()
        full_url = self.config.get('base_url', '') + source_uri
        
        try:
            response = self.session.request(method, full_url, json=data, **kwargs)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reading from API {full_url}: {e}")
            raise IOError(f"API Read Failed: {e}")

    def write(self, data: OutputData, destination_uri: str, method: str = 'POST', **kwargs) -> str:
        """
        Sends data (e.g., inference results or processed features) to an API endpoint.
        """
        if not self.is_connected: self.connect()
        full_url = self.config.get('base_url', '') + destination_uri
        
        try:
            # Typically POST or PUT for writing data
            response = self.session.request(method, full_url, json=data, **kwargs)
            response.raise_for_status()
            logger.info(f"Data successfully written to API endpoint: {full_url}")
            return str(response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error writing to API {full_url}: {e}")
            raise IOError(f"API Write Failed: {e}")

    def close(self):
        """Closes the requests session."""
        if self.session:
            self.session.close()
        super().close()