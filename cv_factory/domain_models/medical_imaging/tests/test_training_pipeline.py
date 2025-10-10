import pytest
from unittest.mock import patch, MagicMock
import os
from domain_models.medical_imaging.pipelines.training_pipeline import run_training_pipeline

# Mock các orchestrator và factories
@pytest.fixture
def mock_training_dependencies():
    with patch("shared_libs.data_ingestion.orchestrator.ingestion_orchestrator.IngestionOrchestrator") as mock_ingestor, \
         patch("shared_libs.ml_core.orchestrators.cv_training_orchestrator.CVTrainingOrchestrator") as mock_orchestrator, \
         patch("domain_models.medical_imaging.pipelines.training_pipeline.create_model_from_config") as mock_model_creator:
        
        mock_ingestor.return_value.run_ingestion.return_value = "mock_raw_data"
        mock_model = MagicMock()
        mock_model_creator.return_value = mock_model
        
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.run.return_value = {"model": mock_model, "metrics": {"accuracy": 0.9}}
        mock_orchestrator.return_value = mock_orchestrator_instance

        yield

@pytest.fixture
def mock_training_config(tmp_path):
    """Create a temporary config file for testing."""
    config_data = """
type: training
pipeline:
  data_ingestion:
    sources:
      - type: "image"
        params:
          source: "./data/train_images/"
  preprocessing:
    pipeline: []
  model:
    name: "resnet18"
    pretrained: true
  trainer:
    type: "cnn"
    params:
      epochs: 1
  evaluator:
    type: "evaluation_orchestrator"
    params:
      task_type: "classification"
      metrics: ["accuracy"]
"""
    config_file = tmp_path / "pipeline_config.yaml"
    config_file.write_text(config_data)
    return str(tmp_path)

def test_training_pipeline_runs_successfully(mock_training_dependencies, mock_training_config):
    """Test that the training pipeline runs without errors."""
    try:
        results = run_training_pipeline(configs_path=mock_training_config)
        assert "model" in results
        assert "metrics" in results
        assert results["metrics"]["accuracy"] == 0.9
    except Exception as e:
        pytest.fail(f"Training pipeline failed unexpectedly: {e}")