# cv_factory/shared_libs/data_ingestion/factories/stream_connector_factory.py (Hardened)

import logging
from typing import Dict, Any, Type

# Import Base Stream Abstraction
from ..base.base_stream_connector import BaseStreamConnector

# Import Concrete Stream Connectors 
from ..connectors.kafka_connector import KafkaConnector
from ..connectors.camera_stream_connector import CameraStreamConnector

logger = logging.getLogger(__name__)

class StreamConnectorFactory:
    """
    Factory class responsible for creating concrete instances of BaseStreamConnector 
    for managing continuous data streams (consume/produce).
    
    This ensures that all stream handling adheres to the standardized stream connector contract.
    """
    
    # Mapping of stream connector types to their concrete class implementations
    CONNECTOR_MAPPING: Dict[str, Type[BaseStreamConnector]] = {
        "kafka": KafkaConnector,
        "camera": CameraStreamConnector,
    }

    @staticmethod
    def get_stream_connector(connector_type: str, connector_config: Dict[str, Any], connector_id: str) -> BaseStreamConnector:
        """
        Creates and returns a concrete stream connector instance.

        Args:
            connector_type: The type of stream connector to create (e.g., 'kafka', 'camera').
            connector_config: Configuration dictionary passed to the connector's constructor.
            connector_id: A unique ID for the connector instance.

        Returns:
            BaseStreamConnector: An instance of the requested concrete stream connector.

        Raises:
            ValueError: If the connector_type is not supported.
            RuntimeError: If stream connector instantiation fails.
        """
        connector_type = connector_type.lower()
        
        if connector_type not in StreamConnectorFactory.CONNECTOR_MAPPING:
            raise ValueError(
                f"Unsupported stream connector type: '{connector_type}'. "
                f"Available types are: {list(StreamConnectorFactory.CONNECTOR_MAPPING.keys())}"
            )
            
        ConnectorClass = StreamConnectorFactory.CONNECTOR_MAPPING[connector_type]
        
        try:
            # Instantiate the stream connector, using 'config' to align with BaseStreamConnector.__init__
            connector = ConnectorClass(
                connector_id=connector_id,
                config=connector_config
            )
            logger.info(f"[{connector_id}] Successfully created {connector_type} stream connector.")
            return connector
        except Exception as e:
            logger.error(f"[{connector_id}] Failed to instantiate {connector_type} stream connector: {e}")
            raise RuntimeError(f"Stream connector creation failed for type '{connector_type}': {e}")