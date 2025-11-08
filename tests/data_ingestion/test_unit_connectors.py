# tests/data_ingestion/test_unit_connectors.py (Excerpt for DICOMConnector)

import pytest
import numpy as np
from pydicom import Dataset
from cv_factory.shared_libs.data_ingestion.connectors.dicom_connector import DICOMConnector
from cv_factory.shared_libs.data_ingestion.utils.dicom_utils import anonymize_dicom_file
from pydicom.tag import Tag

# Mock necessary utility functions and data for isolation
def mock_dicom_to_numpy(dataset):
    """Mocks conversion to a numpy array."""
    return np.zeros((10, 10))

def mock_anonymize_dicom(dataset):
    """Mocks anonymization: removes PatientName tag."""
    if Tag(0x0010, 0x0010) in dataset:
        del dataset[Tag(0x0010, 0x0010)]
    return dataset

@pytest.fixture
def mock_dicom_file(tmp_path):
    """Creates a mock DICOM file with PHI for testing."""
    dcm_file = tmp_path / "test.dcm"
    ds = Dataset()
    ds.PatientName = "John Doe"
    ds.PatientID = "12345"
    ds.Rows = 10
    ds.Columns = 10
    ds.PixelData = b'\x00' * 200  # Mock pixel data
    ds.save_as(str(dcm_file))
    return str(dcm_file)

def test_dicom_read_anonymize_phi(mock_dicom_file, monkeypatch):
    """
    Tests if the DICOMConnector correctly anonymizes PHI during read.
    """
    # Patch the real anonymization and numpy conversion logic
    monkeypatch.setattr('cv_factory.shared_libs.data_ingestion.connectors.dicom_connector.anonymize_dicom', mock_anonymize_dicom)
    monkeypatch.setattr('cv_factory.shared_libs.data_ingestion.connectors.dicom_connector.dicom_to_numpy', mock_dicom_to_numpy)

    connector = DICOMConnector(connector_id="dcm_test", connector_config={})
    
    # 1. Test read without anonymization (returns dataset)
    data_raw = connector.read(mock_dicom_file, as_numpy=False)
    assert data_raw.PatientName == "John Doe"

    # 2. Test read with anonymization (returns dataset)
    data_anon = connector.read(mock_dicom_file, as_numpy=False, anonymize_phi=True)
    assert "PatientName" not in data_anon
    
def test_dicom_read_file_not_found():
    """
    Tests if FileNotFoundError is raised for non-existent DICOM file.
    """
    connector = DICOMConnector(connector_id="dcm_test", connector_config={})
    with pytest.raises(FileNotFoundError):
        connector.read("/non/existent/file.dcm")

# ... other DICOMConnector tests (e.g., test write parquet)