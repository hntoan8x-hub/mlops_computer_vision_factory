import logging
import os
from typing import Dict, Any

from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator
from shared_libs.ml_core.configs.config_utils import load_config, validate_config
from shared_libs.ml_core.configs.evaluator_config_schema import EvaluatorConfig

logger = logging.getLogger(__name__)

def run_evaluation_pipeline(configs_path: str, model: Any, data_loader: Any) -> Dict[str, Any]:
    """
    Executes the model evaluation pipeline.

    Args:
        configs_path (str): The path to the configuration files.
        model (Any): The model to be evaluated.
        data_loader (Any): The data loader for the evaluation dataset.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics and explanations.
    """
    # 1. Load and validate the evaluation configuration
    try:
        config_dict = load_config(os.path.join(configs_path, "evaluation_config.yaml"))
        validated_config = EvaluatorConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading or validating evaluation config: {e}")
        raise

    # 2. Use the main evaluator orchestrator to run the evaluation
    evaluation_orchestrator = EvaluationOrchestrator(config_dict)
    
    results = evaluation_orchestrator.evaluate(model=model, data_loader=data_loader)
    
    # 3. Log metrics to MLflow or a report file
    # This would be part of a larger orchestrator, but we can do it here for demonstration.
    # from shared_libs.ml_core.mlflow_service.implementations.mlflow_logger import MLflowLogger
    # mlflow_logger = MLflowLogger()
    # with mlflow_logger.start_run(run_name="model_evaluation"):
    #     mlflow_logger.log_metrics(results["metrics"])
    
    logger.info("Evaluation pipeline completed successfully.")
    return results