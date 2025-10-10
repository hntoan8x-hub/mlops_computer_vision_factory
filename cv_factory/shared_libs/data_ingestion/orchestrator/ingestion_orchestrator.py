import logging
from typing import Dict, Any, List, Union, Iterator

from shared_libs.data_ingestion.factories.loader_factory import LoaderFactory
from shared_libs.data_ingestion.factories.stream_factory import StreamFactory
from shared_libs.data_ingestion.base.base_loader import BaseLoader, RawData
from shared_libs.data_ingestion.base.base_stream_consumer import BaseStreamConsumer, Frame
from shared_libs.data_ingestion.configs.ingestion_config_schema import IngestionConfig

logger = logging.getLogger(__name__)

class IngestionOrchestrator:
    """
    Orchestrates the data ingestion process based on a configuration.

    This class acts as the main entry point for the ingestion pipeline,
    creating and managing loaders and consumers as defined in the config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator with a validated configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the ingestion configuration.
        """
        self.config = IngestionConfig(**config)
        self.loaders: List[BaseLoader] = []
        self.consumers: List[BaseStreamConsumer] = []
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes loaders and consumers based on the parsed configuration.
        """
        for source_config in self.config.sources:
            source_type = source_config.type
            source_params = source_config.params.dict()

            if source_type in LoaderFactory._LOADER_MAP:
                loader = LoaderFactory.create(source_type, source_params)
                self.loaders.append(loader)
            elif source_type in StreamFactory._CONSUMER_MAP:
                consumer = StreamFactory.create(source_type, source_params)
                self.consumers.append(consumer)
            else:
                logger.warning(f"Unsupported source type '{source_type}' found in config. Skipping.")

    def run_ingestion(self) -> Union[List[RawData], List[Iterator[Frame]]]:
        """
        Executes the ingestion process for all configured sources.

        For each loader, it loads data. For each consumer, it returns the consumer object
        which can then be iterated over.

        Returns:
            Union[List[RawData], List[Iterator[Frame]]]: A list of loaded data objects
                                                         or a list of stream iterators.
        """
        all_data = []

        # Run static loaders
        for loader in self.loaders:
            source = loader.config.get('source')  # Assumes a 'source' key in loader config
            if not source:
                logger.error("Loader config is missing 'source'. Skipping.")
                continue
            try:
                data = loader.load(source)
                all_data.append(data)
                logger.info(f"Successfully loaded data from {source}")
            except Exception as e:
                logger.error(f"Failed to load data from {source}: {e}")
                
        # Return consumers for real-time processing
        if self.consumers:
            all_data.extend(self.consumers)
            
        return all_data