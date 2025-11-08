# cv_factory/shared_libs/data_ingestion/ingestion_orchestrator.py (Hardened)

import logging
from typing import Dict, Any, List, Union, Iterator

# Use unified Connector and Stream Connector Factories
from shared_libs.data_ingestion.factories.connector_factory import ConnectorFactory
from shared_libs.data_ingestion.factories.stream_connector_factory import StreamConnectorFactory

# Use unified Base Connectors
from shared_libs.data_ingestion.base.base_data_connector import BaseDataConnector, OutputData
from shared_libs.data_ingestion.base.base_stream_connector import BaseStreamConnector, StreamData
from shared_libs.data_ingestion.configs.ingestion_config_schema import IngestionConfig, SourceConnector

logger = logging.getLogger(__name__)

# Type alias for clarity
ConnectorUnion = Union[BaseDataConnector, BaseStreamConnector]

class IngestionOrchestrator:
    """
    Orchestrates the data ingestion process based on a configuration.

    This class acts as the main entry point for the ingestion pipeline,
    creating and managing static and stream connectors as defined in the config.

    Attributes:
        config (IngestionConfig): The validated Pydantic configuration object.
        static_connectors (List[BaseDataConnector]): Connectors for static data (Image, Video, DICOM, API).
        stream_connectors (List[BaseStreamConnector]): Connectors for streaming data (Kafka, Camera).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator with a validated configuration.

        Args:
            config: A dictionary containing the ingestion configuration.
        """
        # Hardening: Use Pydantic to validate config immediately and strictly enforce schema
        try:
            self.config = IngestionConfig(**config)
        except Exception as e:
            logger.critical(f"Pydantic Validation Failed for Ingestion Config: {e}")
            raise ValueError(f"Invalid Ingestion Configuration: {e}")
            
        self.static_connectors: List[BaseDataConnector] = []
        self.stream_connectors: List[BaseStreamConnector] = []
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes static and stream connectors based on the parsed configuration.
        """
        for i, source_config in enumerate(self.config.connectors): # Use .connectors from Pydantic schema
            connector_id = f"{source_config.type}_{i}" # Generate unique ID for logging/tracking
            source_type = source_config.type.lower()
            source_params = source_config.dict() # Convert Pydantic model to dict for connector init

            try:
                if source_type in ConnectorFactory.CONNECTOR_MAPPING:
                    connector = ConnectorFactory.get_connector(source_type, source_params, connector_id)
                    self.static_connectors.append(connector)
                elif source_type in StreamConnectorFactory.CONNECTOR_MAPPING:
                    connector = StreamConnectorFactory.get_stream_connector(source_type, source_params, connector_id)
                    self.stream_connectors.append(connector)
                else:
                    logger.warning(f"[{connector_id}] Unsupported source type '{source_type}' found in config. Skipping.")
            except (ValueError, RuntimeError, ConnectionError) as e:
                logger.error(f"[{connector_id}] Failed to initialize connector. Check config/credentials: {e}")

    def run_ingestion(self) -> Union[List[OutputData], List[BaseStreamConnector]]:
        """
        Executes the ingestion process for all configured static sources (read).

        For stream connectors, it establishes the connection and returns the connector instance
        which exposes the `consume()` generator/iterator.

        Returns:
            Union[List[OutputData], List[BaseStreamConnector]]: A list of loaded static data objects 
                                                                or a list of connected stream connectors.
        """
        all_data: List[Union[OutputData, BaseStreamConnector]] = []

        # 1. Run static connectors (Load data)
        for connector in self.static_connectors:
            try:
                # Assuming the primary source is stored in connector.config['uri']
                data = connector.read(source_uri=connector.config.get('uri')) 
                all_data.append(data)
                logger.info(f"[{connector.connector_id}] Successfully read data from {connector.config.get('uri')}")
            except Exception as e:
                logger.error(f"[{connector.connector_id}] Failed to read data from {connector.config.get('uri')}: {e}")
        
        # 2. Connect and return stream connectors (Ready for consumption)
        for connector in self.stream_connectors:
            try:
                connector.connect()
                # Return the connected consumer object for the downstream pipeline to consume
                all_data.append(connector)
                logger.info(f"[{connector.connector_id}] Stream connector connected and ready.")
            except Exception as e:
                logger.error(f"[{connector.connector_id}] Failed to connect stream: {e}")
                # Log error but continue with other connectors
                
        # NOTE: A real-world use case might require separate handling for static/stream data,
        # but adhering to the original function signature, we return a unified list.
        return all_data