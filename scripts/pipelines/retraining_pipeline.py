import logging
import os
from typing import Dict, Any

from shared_libs.ml_core.retraining.orchestrator.retrain_orchestrator import RetrainOrchestrator
from shared_libs.ml_core.retraining.configs.retrain_config_schema import RetrainConfig
from shared_libs.ml_core.configs.config_utils import load_config

logger = logging.getLogger(__name__)

def run_retraining_pipeline(configs_path: str, **kwargs: Dict[str, Any]) -> None:
    """
    Executes the retraining pipeline to check for triggers and submit a new job if needed.

    Args:
        configs_path (str): The path to the configuration files.
        **kwargs: Data required for trigger checks (e.g., drift reports, current metrics).
    """
    # 1. Load and validate the retraining configuration
    try:
        config_dict = load_config(os.path.join(configs_path, "retraining_config.yaml"))
        validated_config = RetrainConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading or validating retraining config: {e}")
        raise

    # 2. Use the main orchestrator to check triggers and submit a new job
    retrain_orchestrator = RetrainOrchestrator(config_dict)
    retrain_orchestrator.run(**kwargs)

    logger.info("Retraining pipeline check completed.")

if __name__ == '__main__':
    # Example usage with dummy data
    CONFIGS_PATH = "./domain_models/medical_imaging/configs/"
    dummy_metrics = {"accuracy": 0.84, "f1_score": 0.83}
    run_retraining_pipeline(configs_path=CONFIGS_PATH, current_metrics=dummy_metrics)