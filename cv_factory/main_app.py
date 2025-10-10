import logging
import os
import argparse

from shared_libs.ml_core.configs.config_utils import load_config
from shared_libs.ml_core.orchestrators.cv_pipeline_factory import CVPipelineFactory
from shared_libs.ml_core.orchestrators.configs.orchestrator_config_schema import OrchestratorConfig

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main(config_path: str):
    """
    Main entry point for the CV Factory application.
    Loads a pipeline configuration and runs the specified orchestrator.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Loading pipeline configuration from {config_path}")
    try:
        config_dict = load_config(config_path)
        validated_config = OrchestratorConfig(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load or validate config: {e}")
        return

    logger.info(f"Creating and running '{validated_config.pipeline_type}' pipeline orchestrator.")
    try:
        pipeline_orchestrator = CVPipelineFactory.create(config_dict)
        pipeline_orchestrator.run()
        logger.info(f"Pipeline '{validated_config.pipeline_type}' finished successfully.")
    except Exception as e:
        logger.error(f"An error occurred while running the pipeline: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Computer Vision MLOps pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="cv_factory/domain_models/medical_imaging/configs/pipeline_config.yaml",
        help="Path to the pipeline configuration file."
    )
    args = parser.parse_args()
    main(args.config)