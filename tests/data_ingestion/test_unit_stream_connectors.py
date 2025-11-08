# tests/data_ingestion/test_unit_stream_connectors.py

import pytest
import time
from unittest.mock import MagicMock, patch
from cv_factory.shared_libs.data_ingestion.connectors.kafka_connector import KafkaConnector
from cv_factory.shared_libs.data_ingestion.connectors.camera_stream_connector import CameraStreamConnector
from confluent_kafka import KafkaException

@pytest.fixture
def mock_kafka_consumer():
    """Mocks the confluent_kafka Consumer object."""
    mock_consumer = MagicMock()
    # Mock successful poll for the first two calls, then return None (no message)
    mock_consumer.poll.side_effect = [
        MagicMock(value=b'{"id": 1}', error=lambda: None),
        MagicMock(value=b'{"id": 2}', error=lambda: None),
        None,
    ]
    return mock_consumer

@patch('cv_factory.shared_libs.data_ingestion.connectors.kafka_connector.Consumer', autospec=True)
def test_kafka_consume_yields_data(MockConsumer, mock_kafka_consumer):
    """
    Tests if KafkaConnector.consume correctly yields JSON data and stops.
    """
    MockConsumer.return_value = mock_kafka_consumer
    config = {'bootstrap_servers': ['localhost:9092'], 'input_topic': 'test'}
    
    with KafkaConnector(connector_id="kfk", config=config) as connector:
        # Mock successful connect (as we only patch Consumer here)
        connector.is_connected = True 
        
        consumed_data = list(connector.consume(max_batches=2))
        
        assert len(consumed_data) == 2
        assert consumed_data[0] == {"id": 1}
        assert consumed_data[1] == {"id": 2}
        
    mock_kafka_consumer.close.assert_called_once() # Check safe close

@patch('cv_factory.shared_libs.data_ingestion.connectors.camera_stream_connector.cv2', autospec=True)
def test_camera_consume_timeout_handling(mock_cv2):
    """
    Tests CameraConnector's robustness against stream hang (timeout).
    """
    # 1. Setup mock capture object
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    # Simulate first frame success, then subsequent failures
    mock_capture.read.side_effect = [(True, "frame_data"), (False, None), (False, None)]
    mock_cv2.VideoCapture.return_value = mock_capture
    mock_cv2.cvtColor.return_value = "RGB_frame"

    connector = CameraStreamConnector(connector_id="cam", config={'source': 0})
    
    # Patch time.time() to simulate elapsed time
    with patch('time.time', side_effect=[0, 1, 6, 7]): # 0s (start), 1s (frame 1), 6s (frame 2 fails + timeout), 7s (after break)
        # 2. Connect and consume with short timeout
        connector.connect()
        # The loop should break after the timeout (5s default, 5s elapsed)
        frames = list(connector.consume(frame_timeout_s=5.0)) 
    
    assert frames == ["RGB_frame"]
    assert connector.is_connected is False # Should be forced disconnected after timeout
    mock_capture.release.assert_called_once()