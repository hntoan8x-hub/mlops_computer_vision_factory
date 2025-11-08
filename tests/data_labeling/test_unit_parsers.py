# tests/data_labeling/test_unit_parsers.py

import pytest
import pandas as pd
from pydantic import ValidationError
from cv_factory.shared_libs.data_labeling.manual_annotation.classification_parser import ClassificationParser
from cv_factory.shared_libs.data_labeling.manual_annotation.detection_parser import DetectionParser
from cv_factory.shared_libs.data_labeling.configs.labeler_config_schema import ClassificationLabelerConfig, DetectionLabelerConfig

# --- MOCK DATA AND CONFIG ---

CLASSIFICATION_CONFIG = {"label_source_uri": "s3://labels.csv", "image_path_column": "file_path", "label_column": "class_name"}
DETECTION_CONFIG = {"label_source_uri": "s3://coco.json", "input_format": "coco_json", "normalize_bbox": False}

# --- CLASSIFICATION PARSER TESTS ---

def test_classification_parser_success():
    """Tests successful parsing and Pydantic validation of ClassificationLabels."""
    parser = ClassificationParser(CLASSIFICATION_CONFIG)
    raw_df = pd.DataFrame({
        "file_path": ["img/1.jpg", "img/2.png"],
        "class_name": ["dog", "cat"]
    })
    
    labels = parser.parse(raw_df)
    
    assert len(labels) == 2
    assert labels[0].label == "dog"
    assert "ValidationError" not in str(labels)

def test_classification_parser_invalid_data_skipped(caplog):
    """Tests if entries with invalid data (e.g., empty label) are logged and skipped."""
    parser = ClassificationParser(CLASSIFICATION_CONFIG)
    raw_df = pd.DataFrame({
        "file_path": ["img/valid.jpg", "img/invalid.jpg"],
        "class_name": ["car", ""] # Empty label should fail Pydantic's constr(min_length=1)
    })
    
    labels = parser.parse(raw_df)
    
    assert len(labels) == 1
    assert "Skipping invalid classification entry" in caplog.text

# --- DETECTION PARSER (COCO) TESTS ---

MOCK_COCO = {
    "images": [{"id": 1, "file_name": "img/1.jpg", "width": 1000, "height": 1000}],
    "annotations": [
        # Valid object (pixel coords)
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100]}, # x, y, w, h
        # Invalid object (w=0, will fail BBox validation)
        {"image_id": 1, "category_id": 2, "bbox": [200, 200, 0, 50]}
    ],
    "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
}

def test_detection_parser_coco_parsing_and_validation(caplog):
    """Tests COCO parsing, BBox conversion, and Pydantic validation."""
    parser = DetectionParser(DETECTION_CONFIG)
    # Mock class map to resolve IDs
    parser.category_id_to_name = {1: "person", 2: "car"} 
    
    labels = parser.parse(MOCK_COCO)
    
    assert len(labels) == 1 # Only one image
    detection_label = labels[0]
    
    # 1. Check normalization (normalize_bbox=False means pixel coords are passed)
    assert detection_label.objects[0].bbox == (10, 10, 110, 110) # COCO [x,y,w,h] -> [x1,y1,x2,y2]
    assert len(detection_label.objects) == 1 # Invalid BBox was filtered by Pydantic's internal rule (x_max <= x_min or y_max <= y_min)
    assert "Skipping invalid detection entry" in caplog.text

def test_detection_parser_bbox_normalization():
    """Tests pixel-to-normalized BBox conversion logic when normalize_bbox=True."""
    config_norm = DETECTION_CONFIG.copy()
    config_norm['normalize_bbox'] = True
    
    parser = DetectionParser(config_norm)
    parser.category_id_to_name = {1: "person"} 
    
    MOCK_COCO_SMALL = {
        "images": [{"id": 1, "file_name": "img/1.jpg", "width": 100, "height": 200}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 50, 40, 50]}, # x=10, y=50, w=40, h=50
        ],
    }
    
    labels = parser.parse(MOCK_COCO_SMALL)
    
    # Pixel BBox: (10, 50, 50, 100)
    # Normalized BBox: (10/100, 50/200, 50/100, 100/200) = (0.1, 0.25, 0.5, 0.5)
    
    bbox = labels[0].objects[0].bbox
    assert pytest.approx(bbox, rel=1e-3) == (0.1, 0.25, 0.5, 0.5)