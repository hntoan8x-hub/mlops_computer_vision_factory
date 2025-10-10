# cv_factory/shared_libs/data_ingestion/base/base_stream_connector.py

import abc
from typing import Any, Dict, Iterator, Union, List, Optional
import logging

logger = logging.getLogger(__name__)

# Define a type hint for a single frame or a batch of frames, 
# which can be either input or output data.
StreamData = Union[Any, List[Any]]

class BaseStreamConnector(abc.ABC):
    """
    Abstract Base Class for Stream Connectors (Producers and Consumers).

    Defines the contract for reading (consuming) data from a continuous stream 
    and writing (producing) processed results back to a stream, ensuring 
    safe resource management via the Context Manager protocol.
    """

    def __init__(self, connector_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Stream Connector.
        
        Args:
            connector_id (str): A unique identifier for this connector instance.
            config (Optional[Dict[str, Any]]): Configuration for stream connection (e.g., brokers, topics).
        """
        self.connector_id = connector_id
        self.config = config if config is not None else {}
        self.is_connected = False
        
    # --- Required BaseDataConnector Methods ---
    # We follow the standard 'connect' and 'close' methods from the Data Connector philosophy.

    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Establishes the connection to the streaming platform (e.g., connecting to Kafka brokers, 
        initializing a camera stream handler).
        
        Returns:
            bool: True if connection is successful.
        """
        self.is_connected = True
        raise NotImplementedError

    @abc.abstractmethod
    def consume(self, **kwargs: Dict[str, Any]) -> Iterator[StreamData]:
        """
        Consumes and yields data from the input stream.
        
        This method must be a generator, using `yield` to return 
        individual frames or batches of raw stream data.

        Args:
            **kwargs: Configuration arguments for the consumer (e.g., starting offset).

        Yields:
            StreamData: A single frame or a batch of data from the stream.
        """
        if not self.is_connected:
            raise ConnectionError(f"[{self.connector_id}] Cannot consume. Connector is not connected.")
        raise NotImplementedError

    @abc.abstractmethod
    def produce(self, data: StreamData, destination_topic: str, **kwargs: Dict[str, Any]):
        """
        Writes/Produces processed data or inference results back to an output stream.

        This method is crucial for real-time MLOps feedback loops and logging.

        Args:
            data (StreamData): The processed data (e.g., inference result, processed frame) to send.
            destination_topic (str): The stream/topic/endpoint where the data should be sent.
            **kwargs: Configuration arguments for the producer (e.g., partitioning key).
        """
        if not self.is_connected:
            raise ConnectionError(f"[{self.connector_id}] Cannot produce. Connector is not connected.")
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Closes the stream connection and safely releases all underlying resources 
        (e.g., closing Kafka producer/consumer sessions, releasing camera handles).
        """
        self.is_connected = False
        logger.info(f"[{self.connector_id}] Stream Connector closed and resources released.")
        raise NotImplementedError
        
    # --- Context Manager Support for safe resource handling ---

    def __enter__(self):
        """Supports the Context Manager protocol."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Supports the Context Manager protocol, ensuring resources are closed."""
        self.close()
        if exc_type:
            logger.error(f"[{self.connector_id}] Exception occurred during stream operation.", 
                         exc_info=(exc_type, exc_val, exc_tb))
            return False # Re-raise the exception
        return True