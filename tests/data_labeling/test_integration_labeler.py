# tests/data_labeling/test_integration_labeler.py

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from cv_factory.shared_libs.data_labeling.labeling_factory import LabelingFactory
from cv_factory.shared_libs.data_labeling.implementations.classification_labeler import ClassificationLabeler
from cv_factory.shared_libs.data_labeling.base_labeler import BaseLabeler
from cv_factory.shared_libs.data_labeling.configs.labeler_config_schema import LabelerConfig

# --- MOCK CONFIGS ---

VALID_CLASSIFICATION_JOB = {
    "task_type": "classification",
    "params": CLASSIFICATION_CONFIG
}

VALID_DETECTION_JOB_AUTO = {
    "task_type": "detection",
    "params": DETECTION_CONFIG,
    "annotation_mode": "auto", # Triggers Auto Annotator init
    "auto_annotation": {"annotator_type": "detection", "model_path": "/m/det.pt", "min_confidence": 0.85}
}

# --- FACTORY INTEGRATION TESTS ---

def test_labeling_factory_success():
    """Tests LabelingFactory successfully validates config and instantiates the correct Labeler."""
    labeler = LabelingFactory.get_labeler("clf_job_1", VALID_CLASSIFICATION_JOB)
    
    assert isinstance(labeler, BaseLabeler)
    assert isinstance(labeler, ClassificationLabeler)
    # Check config validation integrity
    assert labeler.validated_config.task_type == "classification"

def test_labeling_factory_invalid_config_failure():
    """Tests Factory failure on Pydantic config mismatch."""
    INVALID_JOB = VALID_CLASSIFICATION_JOB.copy()
    INVALID_JOB['params']['label_source_uri'] = "" # Fails constr(min_length=5)
    
    with pytest.raises(ValueError, match="failed Pydantic validation"):
        LabelingFactory.get_labeler("invalid_job", INVALID_JOB)

# --- LABELER E2E LOAD TEST (MOCK IO & PARSER) ---

@patch('cv_factory.shared_libs.data_labeling.implementations.classification_labeler.ManualAnnotatorFactory.get_annotator')
@patch('cv_factory.shared_libs.data_labeling.implementations.classification_labeler.ClassificationLabeler.get_source_connector')
def test_classification_labeler_load_labels_e2e_manual(mock_get_connector, mock_get_parser):
    """
    Tests the end-to-end load_labels flow for Manual Classification: 
    Connector -> Parser -> Validation -> Tensor Ready.
    """
    # 1. MOCK Connector Read
    mock_connector = MagicMock()
    mock_connector.read.return_value = pd.DataFrame({"file_path": ["a.jpg"], "class_name": ["dog"]})
    mock_get_connector.return_value.__enter__.return_value = mock_connector

    # 2. MOCK Parser (simulating Pydantic objects output)
    mock_parser = MagicMock()
    # Mock parser output is List[Pydantic Object]
    mock_pydantic_label = MagicMock(label='dog', image_path='a.jpg', model_dump=lambda: {'label': 'dog', 'image_path': 'a.jpg'}) 
    mock_parser.parse.return_value = [mock_pydantic_label]
    mock_get_parser.return_value = mock_parser

    # 3. Instantiate and Load
    labeler = LabelingFactory.get_labeler("clf_test_e2e", VALID_CLASSIFICATION_JOB)
    labels_dict = labeler.load_labels()
    
    # ASSERTIONS
    assert len(labels_dict) == 1
    assert labels_dict[0]['label'] == 'dog'
    
    # Check Tensor conversion
    label_tensor = labeler.convert_to_tensor(labels_dict[0])
    assert label_tensor.item() == 1 # 'dog' is mapped to ID 1 (default map in __init__)

@patch('cv_factory.shared_libs.data_labeling.implementations.detection_labeler.AutoAnnotatorFactory.get_annotator')
@patch('cv_factory.shared_libs.data_labeling.implementations.detection_labeler.DetectionLabeler.get_source_connector')
def test_detection_labeler_load_labels_e2e_auto(mock_get_connector, mock_get_annotator):
    """
    Tests the load_labels flow for Auto Detection: 
    Loads metadata, checks if Auto Annotator is initialized.
    """
    # MOCK Connector Read returns metadata list
    mock_connector = MagicMock()
    mock_connector.read.return_value = [{"image_path": "a.jpg"}, {"image_path": "b.jpg"}]
    mock_get_connector.return_value.__enter__.return_value = mock_connector

    # Instantiate and Load (Auto Mode)
    labeler = LabelingFactory.get_labeler("det_test_e2e_auto", VALID_DETECTION_JOB_AUTO)
    labels_metadata = labeler.load_labels()
    
    # ASSERTIONS
    assert len(labels_metadata) == 2
    assert labeler.annotation_mode == "auto"
    mock_get_annotator.assert_called_once() # Auto Annotator must be initialized