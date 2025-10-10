import pytest
from unittest.mock import patch, MagicMock, Mock
from domain_models.medical_imaging.services.imaging_service_orchestrator import ImagingServiceOrchestrator

@pytest.fixture
def mock_service_config(tmp_path):
    config_data = """
service:
  model_name: "test_model"
  model_version: "Production"
  api_port: 8000
  api_workers: 1
  inference_batch_size: 1
training:
  type: "cnn"
  params:
    epochs: 1
evaluation:
  task_type: "classification"
  metrics: ["accuracy"]
"""
    config_file = tmp_path / "service_config.yaml"
    config_file.write_text(config_data)
    return str(tmp_path)

@pytest.fixture
def mock_service_dependencies():
    with patch("domain_models.medical_imaging.services.imaging_predictor.ImagingPredictor") as mock_predictor, \
         patch("domain_models.medical_imaging.services.imaging_trainer.ImagingTrainer") as mock_trainer, \
         patch("domain_models.medical_imaging.services.imaging_evaluator.ImagingEvaluator") as mock_evaluator:
        
        mock_predictor.return_value.predict.return_value = "mock_prediction"
        mock_trainer.return_value.run_training.return_value = {"metrics": {"accuracy": 0.9}}
        mock_evaluator.return_value.run_evaluation.return_value = {"metrics": {"accuracy": 0.95}}

        yield

def test_service_orchestrator_initializes(mock_service_dependencies, mock_service_config):
    """Test that the service orchestrator initializes all components correctly."""
    orchestrator = ImagingServiceOrchestrator(configs_path=mock_service_config)
    assert orchestrator.predictor is not None
    assert orchestrator.trainer is not None
    assert orchestrator.evaluator is not None
    
def test_service_orchestrator_calls_predict(mock_service_dependencies, mock_service_config):
    """Test that the orchestrator's predict method correctly calls the predictor."""
    orchestrator = ImagingServiceOrchestrator(configs_path=mock_service_config)
    result = orchestrator.predict(input_data={"image_id": "test"})
    assert result == "mock_prediction"
    orchestrator.predictor.predict.assert_called_once()
    
def test_service_orchestrator_calls_train(mock_service_dependencies, mock_service_config):
    """Test that the orchestrator's train method correctly calls the trainer."""
    orchestrator = ImagingServiceOrchestrator(configs_path=mock_service_config)
    result = orchestrator.train_model(data_path="mock_path")
    assert result["metrics"]["accuracy"] == 0.9
    orchestrator.trainer.run_training.assert_called_once()