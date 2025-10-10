import pytest
from pydantic import ValidationError
from domain_models.medical_imaging.schemas.input_schema import MedicalImageInput
from domain_models.medical_imaging.schemas.output_schema import PredictionOutput, BoundingBox

def test_medical_image_input_schema_valid():
    """Test a valid medical image input."""
    valid_data = {
        "image_id": "img_001",
        "patient_id": "patient_123",
        "modality": "CT",
        "image_data": b"fake_image_data"
    }
    input_obj = MedicalImageInput(**valid_data)
    assert input_obj.image_id == "img_001"
    assert input_obj.modality == "CT"

def test_medical_image_input_schema_invalid():
    """Test an invalid medical image input (missing required field)."""
    invalid_data = {
        "image_id": "img_002",
        "modality": "MRI",
        "image_data": b"fake_image_data"
    }
    with pytest.raises(ValidationError):
        MedicalImageInput(**invalid_data)

def test_prediction_output_schema_valid_detection():
    """Test a valid detection output prediction."""
    valid_data = {
        "image_id": "img_003",
        "patient_id": "patient_123",
        "prediction_time_ms": 150.5,
        "detections": [
            {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60, "class_name": "lesion", "confidence": 0.95}
        ]
    }
    output_obj = PredictionOutput(**valid_data)
    assert len(output_obj.detections) == 1
    assert output_obj.detections[0].class_name == "lesion"