# cv_factory/shared_libs/data_ingestion/ingestion_orchestrator.py (HARDENED FOR LIVE DATA OVERRIDE)

import logging
from typing import Dict, Any, List, Union, Iterator, Optional

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
    
    HARDENED: Now supports dynamic overriding of connector URIs for live/production data access.

    Attributes:
        config (IngestionConfig): The validated Pydantic configuration object.
        static_connectors (List[BaseDataConnector]): Connectors for static data (Image, Video, DICOM, API).
        stream_connectors (List[BaseStreamConnector]): Connectors for streaming data (Kafka, Camera).
    """

    def __init__(self, 
                 config: Dict[str, Any], 
                 # <<< NEW: Tham số tùy chọn để ghi đè URI dữ liệu Production >>>
                 live_data_uri_override: Optional[str] = None):
        """
        Initializes the orchestrator with a validated configuration.

        Args:
            config: A dictionary containing the ingestion configuration.
            live_data_uri_override: URI để ghi đè nguồn dữ liệu cho ngữ cảnh Production (Health Check).
        """
        try:
            self.config = IngestionConfig(**config)
        except Exception as e:
            logger.critical(f"Pydantic Validation Failed for Ingestion Config: {e}")
            raise ValueError(f"Invalid Ingestion Configuration: {e}")
            
        self.static_connectors: List[BaseDataConnector] = []
        self.stream_connectors: List[BaseStreamConnector] = []
        self.live_data_uri_override = live_data_uri_override
        
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes static and stream connectors based on the parsed configuration.
        """
        for i, source_config in enumerate(self.config.connectors):
            connector_id = f"{source_config.type}_{i}"
            source_type = source_config.type.lower()
            source_params = source_config.dict()

            # --- HARDENING LOGIC: Override Live Data URI ---
            if self.live_data_uri_override and source_params.get('context') == 'production':
                old_uri = source_params.get('uri')
                source_params['uri'] = self.live_data_uri_override
                logger.warning(f"[{connector_id}] Production URI overridden: {old_uri} -> {self.live_data_uri_override}")
            # --- END HARDENING LOGIC ---

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
        """
        all_data: List[Union[OutputData, BaseStreamConnector]] = []

        # 1. Run static connectors (Load data)
        for connector in self.static_connectors:
            try:
                # Assuming the primary source is stored in connector.config['uri']
                # Tái cấu trúc để sử dụng config.get('uri') đã được override nếu cần
                data = connector.read(source_uri=connector.config.get('uri')) 
                all_data.append(data)
                logger.info(f"[{connector.connector_id}] Successfully read data from {connector.config.get('uri')}")
            except Exception as e:
                logger.error(f"[{connector.connector_id}] Failed to read data from {connector.config.get('uri')}: {e}")
        
        # 2. Connect and return stream connectors (Ready for consumption)
        for connector in self.stream_connectors:
            try:
                connector.connect()
                all_data.append(connector)
                logger.info(f"[{connector.connector_id}] Stream connector connected and ready.")
            except Exception as e:
                logger.error(f"[{connector.connector_id}] Failed to connect stream: {e}")
                
        return all_data