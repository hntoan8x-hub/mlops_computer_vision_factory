# cv_factory/shared_libs/data_ingestion/base/base_data_connector.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Standardizing the expected output type for CV data (images, video streams, metadata)
OutputData = Union[Any, List[Any], Dict[str, Any]]

class BaseDataConnector(ABC):
    """
    Abstract Base Class (ABC) for all Data Connectors.
    
    Defines the contract (read, write, connect) to abstract data access 
    and persistence, adhering to the Single Responsibility Principle (SRP).
    """

    def __init__(self, connector_id: str, connector_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseDataConnector.
        
        Args:
            connector_id (str): A unique identifier for this connector instance.
            connector_config (Optional[Dict[str, Any]]): Specific configuration for the connection.
        """
        self.connector_id = connector_id
        self.config = connector_config if connector_config is not None else {}
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Establishes a physical or logical connection to the data source/destination.
        
        Raises:
            ConnectionError: If the connection cannot be established.
            
        Returns:
            bool: True if the connection is successful.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, source_uri: str, **kwargs) -> OutputData:
        """
        Reads data from the specified source (URI, path, query, topic).
        
        This is the core method for loading raw or pre-normalized data.
        
        Args:
            source_uri (str): The URI, path, or identifier of the data source.
            **kwargs: Optional custom parameters (e.g., chunk_size, query_filters).
            
        Returns:
            OutputData: The loaded data (e.g., list of image arrays, DICOM objects, metadata table).
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, data: OutputData, destination_uri: str, **kwargs) -> str:
        """
        Writes/Persists data to the destination (Data Lake, Feature Store, Cache, etc.).
        
        This method is crucial for storing pre-processed data or generated features 
        required for MLOps tracking and future training runs.
        
        Args:
            data (OutputData): The data object to write.
            destination_uri (str): The URI, path, or identifier of the storage destination.
            **kwargs: Optional custom parameters (e.g., compression, file_format).
            
        Returns:
            str: The final path/URI of the written data (for tracking/versioning).
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Closes the connection and releases resources if necessary.
        """
        self.is_connected = False
        # The implementation in subclasses should handle the actual resource closing
        # print(f"[{self.connector_id}] Connection closed.")

    def __enter__(self):
        """Supports the Context Manager protocol (entering the 'with' block)."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Supports the Context Manager protocol (exiting the 'with' block)."""
        self.close()