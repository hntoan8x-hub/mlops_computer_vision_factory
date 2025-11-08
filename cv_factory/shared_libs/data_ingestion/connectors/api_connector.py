# cv_factory/shared_libs/data_ingestion/connectors/api_connector.py

import logging
import requests
from typing import Dict, Any, Union, Optional
from ..base.base_data_connector import BaseDataConnector, OutputData

logger = logging.getLogger(__name__)

class APIConnector(BaseDataConnector):
    """
    Concrete connector for fetching data from and sending data to REST API endpoints.

    The connector utilizes the requests library's Session for persistent connections 
    and handles standard HTTP request methods (GET, POST, PUT, etc.).
    """

    def connect(self) -> bool:
        """
        Connect for API means initializing a session and setting common headers (e.g., authentication).

        Returns:
            bool: True if the API Connector session is successfully established.
        """
        self.session = requests.Session()
        # Production Hardening: Apply authentication headers from config here if available
        # if token := self.config.get("token"):
        #     self.session.headers.update({'Authorization': f'Bearer {token}'})
        
        self.is_connected = True
        logger.info(f"[{self.connector_id}] API Connector session established.")
        return True

    def read(self, source_uri: str, method: str = 'GET', data: Optional[Dict[str, Any]] = None, **kwargs) -> OutputData:
        """
        Fetches data from an API endpoint. Supports GET or POST methods for reading.

        Args:
            source_uri: The API path relative to the base URL (e.g., '/images/list').
            method: The HTTP method to use (e.g., 'GET', 'POST'). Defaults to 'GET'.
            data: JSON payload for POST/PUT requests (optional).
            **kwargs: Optional custom parameters, including Pydantic-validated 'timeout_seconds'.

        Returns:
            OutputData: The parsed JSON response content.

        Raises:
            IOError: If the API request fails (network, timeout, or HTTP error).
        """
        if not self.is_connected: self.connect()
        full_url = self.config.get('base_url', '') + source_uri
        
        # Hardening: Use validated timeout from config, fallback to 10 seconds
        timeout = kwargs.pop('timeout_seconds', self.config.get('timeout_seconds', 10))

        try:
            response = self.session.request(
                method, 
                full_url, 
                json=data, 
                timeout=timeout, # Apply timeout
                **kwargs
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Robustness: Check content type before parsing JSON
            if 'application/json' in response.headers.get('Content-Type', ''):
                return response.json()
            
            # If not JSON, return raw text/content
            return response.content

        except requests.exceptions.Timeout as e:
            logger.error(f"API request timed out to {full_url} after {timeout}s: {e}")
            raise IOError(f"API Read Failed (Timeout): {e}")
        except requests.exceptions.RequestException as e:
            # Catch all other requests errors (ConnectionError, HTTPError, TooManyRedirects, etc.)
            logger.error(f"Error reading from API {full_url}. Status: {getattr(e.response, 'status_code', 'N/A')}. Details: {e}")
            raise IOError(f"API Read Failed: {e}")

    def write(self, data: OutputData, destination_uri: str, method: str = 'POST', **kwargs) -> str:
        """
        Sends data (e.g., inference results or processed features) to an API endpoint.

        Args:
            data: The data payload to be sent (will be serialized as JSON).
            destination_uri: The API path relative to the base URL (e.g., '/results/log').
            method: The HTTP method to use (e.g., 'POST', 'PUT'). Defaults to 'POST'.
            **kwargs: Optional custom parameters, including Pydantic-validated 'timeout_seconds'.

        Returns:
            str: The final HTTP status code or a success message.

        Raises:
            IOError: If the API request fails.
        """
        if not self.is_connected: self.connect()
        full_url = self.config.get('base_url', '') + destination_uri
        
        # Hardening: Use validated timeout
        timeout = kwargs.pop('timeout_seconds', self.config.get('timeout_seconds', 10))
        
        try:
            # Typically POST or PUT for writing data
            response = self.session.request(
                method, 
                full_url, 
                json=data, 
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            logger.info(f"Data successfully written to API endpoint: {full_url}. Status: {response.status_code}")
            return str(response.status_code)
        except requests.exceptions.Timeout as e:
            logger.error(f"API write request timed out to {full_url} after {timeout}s: {e}")
            raise IOError(f"API Write Failed (Timeout): {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error writing to API {full_url}. Status: {getattr(e.response, 'status_code', 'N/A')}. Details: {e}")
            raise IOError(f"API Write Failed: {e}")

    def close(self):
        """
        Closes the requests session and releases resources.
        """
        if self.session:
            self.session.close()
        super().close()