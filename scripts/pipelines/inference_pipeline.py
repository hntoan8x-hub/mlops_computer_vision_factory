import logging
import os
import numpy as np
from typing import Dict, Any, Union, List

from shared_libs.ml_core.orchestrators.cv_inference_orchestrator import CVInferenceOrchestrator
from shared_libs.ml_core.orchestrators.configs.orchestrator_config_schema import InferenceConfig
from shared_libs.ml_core.configs.config_utils import load_config, validate_config

logger = logging.getLogger(__name__)

def run_inference_pipeline(configs_path: str, raw_input_data: Union[np.ndarray, List[np.ndarray]]) -> List[Dict[str, Any]]:
    """
    Executes the inference pipeline for a medical imaging model.

    Args:
        configs_path (str): The path to the configuration files.
        raw_input_data (Union[np.ndarray, List[np.ndarray]]): The raw input image(s) to predict.

    Returns:
        List[Dict[str, Any]]: A list of prediction results.
    """
    # 1. Load and validate the master configuration file for the pipeline
    try:
        config_dict = load_config(os.path.join(configs_path, "pipeline_config.yaml"))
        validated_config = validate_config(config_dict)
    except Exception as e:
        logger.error(f"Error loading or validating pipeline config: {e}")
        raise

    # 2. Use the main orchestrator to run the entire pipeline
    if validated_config.pipeline_type != "inference":
        raise ValueError("The provided config is not for an inference pipeline.")

    inference_orchestrator = CVInferenceOrchestrator(config_dict)

    # The run() method handles all steps: preprocessing -> prediction -> post-processing
    predictions = inference_orchestrator.run(inputs=raw_input_data)

    logger.info("Inference pipeline completed successfully.")
    return predictions

if __name__ == '__main__':
    # Example usage with dummy data
    CONFIGS_PATH = "./domain_models/medical_imaging/configs/"
    dummy_image = np.random.rand(512, 512, 1).astype(np.float32)
    predictions = run_inference_pipeline(configs_path=CONFIGS_PATH, raw_input_data=dummy_image)
    print(predictions)