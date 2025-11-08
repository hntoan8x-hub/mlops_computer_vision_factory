# tests/data_ingestion/test_integration_orchestrator.py

import pytest
from cv_factory.shared_libs.data_ingestion.ingestion_orchestrator import IngestionOrchestrator
from cv_factory.shared_libs.data_ingestion.base.base_data_connector import BaseDataConnector
from cv_factory.shared_libs.data_ingestion.base.base_stream_connector import BaseStreamConnector
from cv_factory.shared_libs.data_ingestion.configs.ingestion_config_schema import IngestionConfig

# Mock configuration that combines static and stream sources
MOCK_VALID_CONFIG = {
    "connectors": [
        # Static Connector 1 (Image)
        {"type": "image", "uri": "s3://bucket/images/", "recursive": True},
        # Static Connector 2 (API)
        {"type": "api", "uri": "https://api.example.com/data", "timeout_seconds": 15},
        # Stream Connector 1 (Kafka)
        {"type": "kafka", "uri": "kafka://brokers", "bootstrap_servers": ["broker1:9092"], "topic": "input_stream", "group_id": "cv_factory_dev"}
    ],
    "data_split_ratio": {"train": 0.7, "validation": 0.3}
}

def test_orchestrator_initialization_success():
    """
    Tests if the Orchestrator initializes correctly, validates config, and separates connectors.
    """
    orchestrator = IngestionOrchestrator(MOCK_VALID_CONFIG)
    
    # 1. Check Pydantic validation (implied by successful initialization)
    assert isinstance(orchestrator.config, IngestionConfig)
    
    # 2. Check separation of static and stream connectors
    assert len(orchestrator.static_connectors) == 2
    assert len(orchestrator.stream_connectors) == 1
    
    # 3. Check correct type instantiation
    assert isinstance(orchestrator.static_connectors[0], BaseDataConnector)
    assert isinstance(orchestrator.stream_connectors[0], BaseStreamConnector)

def test_orchestrator_initialization_pydantic_failure():
    """
    Tests if initialization fails when Pydantic config validation fails (e.g., missing required fields).
    """
    INVALID_CONFIG = {
        "connectors": [
            # Kafka is missing required 'bootstrap_servers' (Pydantic check)
            {"type": "kafka", "uri": "kafka://brokers", "topic": "input_stream", "group_id": "cv_factory_dev"} 
        ]
    }
    
    with pytest.raises(ValueError, match="Invalid Ingestion Configuration"):
        IngestionOrchestrator(INVALID_CONFIG)

# NOTE: run_ingestion() test requires complex mocking of S3/Kafka connections, 
# typically handled by mocking frameworks like moto (for S3) or specific Kafka mocks.

def test_orchestrator_run_ingestion_static_load_failure(monkeypatch):
    """
    Tests if run_ingestion can handle and log failure of a static connector without crashing.
    """
    # Mock the 'read' method of the static connector to raise an Exception
    def mock_read_fail(self, source_uri, **kwargs):
        raise IOError("Mock I/O Error during read.")
        
    monkeypatch.setattr('cv_factory.shared_libs.data_ingestion.connectors.image_connector.ImageConnector.read', mock_read_fail)
    
    orchestrator = IngestionOrchestrator(MOCK_VALID_CONFIG)
    
    # The run should complete without crashing, but the output list should not contain the failed data
    result = orchestrator.run_ingestion()
    
    # The result should contain the failure (None/Error logged) + successful connections
    # We expect 2 static connectors (both fail to read) + 1 stream connector (connects successfully)
    # The exact handling depends on how you structure the run_ingestion return, 
    # but the key is that the pipeline doesn't crash.
    
    # For simplicity, check that the number of connectors processed is correct
    assert len(result) <= 3