import pytest
from unittest.mock import patch, MagicMock
import os
import numpy as np
from domain_models.medical_imaging.pipelines.inference_pipeline import run_inference_pipeline

@pytest.fixture
def mock_inference_dependencies():
    with patch("shared_libs.ml_core.orchestrators.cv_inference_orchestrator.CVInferenceOrchestrator") as mock_orchestrator:
        
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.run.return_value = [{"prediction": "lesion"}]
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        yield

@pytest.fixture
def mock_inference_config(tmp_path):
    config_data = """
type: inference
pipeline:
  model:
    name: "resnet18"
    version: "1.0"
  preprocessing:
    pipeline: []
"""
    config_file = tmp_path / "pipeline_config.yaml"
    config_file.write_text(config_data)
    return str(tmp_path)

def test_inference_pipeline_runs_successfully(mock_inference_dependencies, mock_inference_config):
    """Test that the inference pipeline runs and returns a valid prediction."""
    dummy_input = np.random.rand(224, 224, 3)
    predictions = run_inference_pipeline(configs_path=mock_inference_config, raw_input_data=dummy_input)
    assert len(predictions) == 1
    assert predictions[0]["prediction"] == "lesion"