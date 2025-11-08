# cv_factory/shared_libs/data_ingestion/base/base_data_connector.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Standardizing the expected output type for CV data (images, video streams, metadata)
OutputData = Union[Any, List[Any], Dict[str, Any]]

class BaseDataConnector(ABC):
    """
    Abstract Base Class (ABC) for all Data Connectors.
    
    Defines the contract (read, write, connect, close) to abstract data access 
    and persistence, adhering to the Single Responsibility Principle (SRP). 
    It also implements the context manager protocol for safe resource handling.
    
    Attributes:
        connector_id (str): A unique identifier for the connector instance.
        config (Dict[str, Any]): Configuration settings for the connection.
        is_connected (bool): The current connection status.
    """

    def __init__(self, connector_id: str, connector_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseDataConnector.
        
        Args:
            connector_id: A unique identifier for this connector instance.
            connector_config: Specific configuration for the connection.
        """
        self.connector_id = connector_id
        self.config = connector_config if connector_config is not None else {}
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Establishes a physical or logical connection to the data source/destination.
        
        Subclasses must set `self.is_connected = True` upon successful connection.

        Returns:
            bool: True if the connection is successful.

        Raises:
            ConnectionError: If the connection cannot be established.
        """
        self.is_connected = True # Placeholder for contract enforcement
        raise NotImplementedError

    @abstractmethod
    def read(self, source_uri: str, **kwargs) -> OutputData:
        """
        Reads data from the specified source (URI, path, query, topic).
        
        This is the core method for loading raw or pre-normalized data.
        
        Args:
            source_uri: The URI, path, or identifier of the data source.
            **kwargs: Optional custom parameters (e.g., chunk_size, query_filters).
            
        Returns:
            OutputData: The loaded data (e.g., list of image arrays, metadata table).

        Raises:
            RuntimeError: If the connector is not connected when read is called.
        """
        if not self.is_connected:
            raise RuntimeError(f"[{self.connector_id}] Cannot read. Connector is not connected. Call connect() first.")
        raise NotImplementedError

    @abstractmethod
    def write(self, data: OutputData, destination_uri: str, **kwargs) -> str:
        """
        Writes/Persists data to the destination (Data Lake, Feature Store, Cache, etc.).
        
        This method is crucial for storing pre-processed data or generated features.
        
        Args:
            data: The data object to write.
            destination_uri: The URI, path, or identifier of the storage destination.
            **kwargs: Optional custom parameters (e.g., compression, file_format).
            
        Returns:
            str: The final path/URI of the written data (for tracking/versioning).

        Raises:
            RuntimeError: If the connector is not connected when write is called.
        """
        if not self.is_connected:
            raise RuntimeError(f"[{self.connector_id}] Cannot write. Connector is not connected. Call connect() first.")
        raise NotImplementedError

    def close(self):
        """
        Closes the connection and safely sets the connection state.
        
        Subclasses should implement the logic to release resources (e.g., file handles, 
        client sessions) before calling the super().close() method.
        """
        if self.is_connected:
            self.is_connected = False
            logger.info(f"[{self.connector_id}] Connection state set to closed.")

    # --- Context Manager Support (Hardening for safe resource handling) ---

    def __enter__(self):
        """
        Supports the Context Manager protocol (entering the 'with' block).
        Ensures connection is established upon entry.
        
        Returns:
            BaseDataConnector: The connector instance.
        """
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Supports the Context Manager protocol (exiting the 'with' block).
        Ensures resources are closed safely, even if an exception occurs.

        Returns:
            bool: False if the exception should be re-raised (default behavior).
        """
        self.close()
        # If an exception occurred, log it before re-raising
        if exc_type:
            logger.error(f"[{self.connector_id}] Exception occurred during connector operation.", 
                         exc_info=(exc_type, exc_val, exc_tb))
            return False 
        return True