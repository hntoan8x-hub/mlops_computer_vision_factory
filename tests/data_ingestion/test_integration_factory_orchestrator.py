# tests/data_ingestion/test_integration_factory_orchestrator.py

import pytest
from cv_factory.shared_libs.data_ingestion.factories.connector_factory import ConnectorFactory
from cv_factory.shared_libs.data_ingestion.factories.stream_connector_factory import StreamConnectorFactory
from cv_factory.shared_libs.data_ingestion.connectors.image_connector import ImageConnector
from cv_factory.shared_libs.data_ingestion.connectors.kafka_connector import KafkaConnector
from pydantic import ValidationError

def test_connector_factory_creation_success():
    """
    Tests if ConnectorFactory successfully creates a BaseDataConnector instance.
    """
    config = {"uri": "local/path/img.png", "type": "image", "aws_enabled": False}
    connector = ConnectorFactory.get_connector("image", config, "img_01")
    
    assert isinstance(connector, ImageConnector)
    assert connector.connector_id == "img_01"
    assert connector.config['aws_enabled'] is False

def test_connector_factory_unsupported_type_failure():
    """
    Tests if ConnectorFactory raises ValueError for unsupported types.
    """
    with pytest.raises(ValueError, match="Unsupported connector type"):
        ConnectorFactory.get_connector("unsupported_db", {}, "db_01")

def test_stream_factory_creation_with_valid_config():
    """
    Tests if StreamConnectorFactory successfully creates a BaseStreamConnector instance (Kafka).
    """
    config = {"uri": "kafka://brokers", "bootstrap_servers": ["broker1:9092"], "topic": "input"}
    connector = StreamConnectorFactory.get_stream_connector("kafka", config, "kfk_01")
    
    assert isinstance(connector, KafkaConnector)
    assert connector.config['topic'] == 'input'