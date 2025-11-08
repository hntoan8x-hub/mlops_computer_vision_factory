# cv_factory/shared_libs/data_ingestion/factories/connector_factory.py (Hardened)

import logging
from typing import Dict, Any, Type

# Import Base Abstraction
from ..base.base_data_connector import BaseDataConnector

# Import Concrete Connectors (Static Loaders)
from ..connectors.image_connector import ImageConnector
from ..connectors.dicom_connector import DICOMConnector
from ..connectors.video_connector import VideoConnector
from ..connectors.api_connector import APIConnector 

logger = logging.getLogger(__name__)

class ConnectorFactory:
    """
    Factory class responsible for creating concrete instances of BaseDataConnector 
    for static data access (read/write non-streaming data).
    
    This centralizes the creation logic and decouples clients from concrete connector classes.
    """
    
    # Mapping of static connector types to their concrete class implementations
    CONNECTOR_MAPPING: Dict[str, Type[BaseDataConnector]] = {
        "image": ImageConnector,
        "dicom": DICOMConnector,
        "video": VideoConnector,
        "api": APIConnector,
    }

    @staticmethod
    def get_connector(connector_type: str, connector_config: Dict[str, Any], connector_id: str) -> BaseDataConnector:
        """
        Creates and returns a concrete static data connector instance.

        Args:
            connector_type: The type of connector to create (e.g., 'image', 'dicom').
            connector_config: Configuration dictionary passed to the connector's constructor.
            connector_id: A unique ID for the connector instance.

        Returns:
            BaseDataConnector: An instance of the requested concrete connector.

        Raises:
            ValueError: If the connector_type is not supported.
            RuntimeError: If connector instantiation fails.
        """
        connector_type = connector_type.lower()
        
        if connector_type not in ConnectorFactory.CONNECTOR_MAPPING:
            raise ValueError(
                f"Unsupported connector type: '{connector_type}'. "
                f"Available types are: {list(ConnectorFactory.CONNECTOR_MAPPING.keys())}"
            )
            
        ConnectorClass = ConnectorFactory.CONNECTOR_MAPPING[connector_type]
        
        try:
            # Instantiate the connector, passing config and ID
            # Note: Using 'connector_config' for static connectors to match the base class expected name
            connector = ConnectorClass(
                connector_id=connector_id,
                connector_config=connector_config
            )
            logger.info(f"[{connector_id}] Successfully created {connector_type} connector.")
            return connector
        except Exception as e:
            logger.error(f"[{connector_id}] Failed to instantiate {connector_type} connector: {e}")
            raise RuntimeError(f"Connector creation failed for type '{connector_type}': {e}")