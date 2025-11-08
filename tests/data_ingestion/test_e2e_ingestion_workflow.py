# tests/data_ingestion/test_e2e_ingestion_workflow.py

import pytest
from unittest.mock import patch, MagicMock
from cv_factory.shared_libs.data_ingestion.ingestion_orchestrator import IngestionOrchestrator
from cv_factory.shared_libs.data_ingestion.base.base_stream_connector import BaseStreamConnector
from cv_factory.shared_libs.data_ingestion.base.base_data_connector import BaseDataConnector
import numpy as np

# MOCK CONFIGURATION
E2E_CONFIG = {
    "connectors": [
        # Static: Image (Success expected)
        {"type": "image", "uri": "s3://bucket/test.jpg", "aws_enabled": True}, 
        # Static: DICOM (Failure expected - should not crash pipeline)
        {"type": "dicom", "uri": "/local/dcm/corrupted.dcm", "anonymize_phi": True}, 
        # Stream: Kafka (Success expected - should return connected object)
        {"type": "kafka", "uri": "kafka://brokers", "bootstrap_servers": ["b1:9092"], "topic": "live"}
    ]
}

@patch('cv_factory.shared_libs.data_ingestion.connectors.kafka_connector.KafkaConnector.connect')
@patch('cv_factory.shared_libs.data_ingestion.connectors.dicom_connector.DICOMConnector.read')
@patch('cv_factory.shared_libs.data_ingestion.connectors.image_connector.ImageConnector.read')
def test_e2e_ingestion_orchestration(mock_image_read, mock_dicom_read, mock_kafka_connect):
    """
    Tests the full orchestration flow with mocked I/O results.
    - Image read succeeds and returns data.
    - DICOM read fails and logs error.
    - Kafka connects successfully and returns the connector object.
    """
    # MOCK BEHAVIOR
    # 1. Image read success
    mock_image_read.return_value = np.ones((100, 100, 3)) 
    
    # 2. DICOM read failure
    mock_dicom_read.side_effect = IOError("Corrupted DICOM file") 
    
    # 3. Kafka connect success (returns None, we check the actual object)
    mock_kafka_connect.return_value = True 

    orchestrator = IngestionOrchestrator(E2E_CONFIG)
    
    # EXECUTE
    results = orchestrator.run_ingestion()

    # ASSERTIONS (Check final state and returned types)
    
    # 1. Total items returned: 2 (Image Data) + 1 (Kafka Connector) = 3
    assert len(results) == 3
    
    # 2. Image result check (Static success)
    image_result = next(r for r in results if isinstance(r, np.ndarray))
    assert image_result.shape == (100, 100, 3)
    
    # 3. DICOM result check (Static failure)
    # Since DICOM read failed, the Orchestrator should skip adding the result.
    # We check that only the expected types are present.
    assert not any(r for r in results if r is None) # No explicit None should be returned from failure
    
    # 4. Kafka result check (Stream success)
    kafka_connector = next(r for r in results if isinstance(r, BaseStreamConnector))
    assert isinstance(kafka_connector, BaseStreamConnector)
    mock_kafka_connect.assert_called_once() # Check if connect() was called
    
    # 5. Check call counts to verify sequencing and failure handling
    mock_image_read.assert_called_once()
    mock_dicom_read.assert_called_once() 
    # The pipeline successfully continued past the DICOM failure.