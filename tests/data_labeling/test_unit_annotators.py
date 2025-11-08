# tests/data_labeling/test_unit_annotators.py (Excerpt)

import pytest
import numpy as np
from cv_factory.shared_libs.data_labeling.auto_annotation.detection_proposal import DetectionProposalAnnotator
from cv_factory.shared_libs.data_labeling.auto_annotation.classification_proposal import ClassificationProposalAnnotator

DUMMY_IMAGE_DATA = np.zeros((200, 300, 3), dtype=np.uint8)
DUMMY_METADATA = {"image_path": "/test/img.png", "image_data": DUMMY_IMAGE_DATA}
AUTO_CONFIG = {"model_path": "/mock/model.pt", "min_confidence": 0.8}

# --- CLASSIFICATION ANNOTATOR TESTS ---

def test_classification_annotator_threshold_filter():
    """Tests if labels below min_confidence are filtered out."""
    annotator = ClassificationProposalAnnotator(AUTO_CONFIG)
    
    # Mock inference to return a low confidence result
    with patch.object(annotator, '_run_inference', return_value=("dog", 0.75)):
        labels = annotator.annotate(DUMMY_METADATA)
        assert len(labels) == 0 # 0.75 < 0.8 threshold

def test_classification_annotator_success():
    """Tests successful classification annotation and Pydantic output."""
    annotator = ClassificationProposalAnnotator(AUTO_CONFIG)
    
    # Mock inference to return a high confidence result
    with patch.object(annotator, '_run_inference', return_value=("cat", 0.95)):
        labels = annotator.annotate(DUMMY_METADATA)
        
        assert len(labels) == 1
        assert labels[0].label == "cat"
        assert labels[0].image_path == "/test/img.png"

# --- DETECTION ANNOTATOR TESTS ---

def test_detection_annotator_pydantic_bbox_validation(caplog):
    """
    Tests if the annotator filters out semantically invalid BBoxes 
    (e.g., x_max <= x_min) using Pydantic Validation.
    """
    annotator = DetectionProposalAnnotator(AUTO_CONFIG)
    
    # Mock inference to return: [Valid, Invalid (x_max < x_min), Low Confidence]
    W, H = DUMMY_IMAGE_DATA.shape[1], DUMMY_IMAGE_DATA.shape[0]
    raw_predictions = [
        # Valid BBox
        ((10, 10, 50, 50), "person", 0.9), 
        # Invalid BBox (x_min > x_max)
        ((80, 80, 70, 90), "invalid", 0.9), 
        # Low Confidence BBox (0.7 < 0.8 threshold)
        ((100, 100, 150, 150), "filtered", 0.7) 
    ]
    
    with patch.object(annotator, '_run_inference', return_value=raw_predictions):
        labels = annotator.annotate(DUMMY_METADATA)
        
        # Only the first (valid, high confidence) object should pass
        assert len(labels) == 1
        assert len(labels[0].objects) == 1
        assert "Invalid Detection Object filtered" in caplog.text