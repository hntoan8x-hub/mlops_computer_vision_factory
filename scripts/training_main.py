# cv_factory/scripts/training_main.py

import logging
import os
import sys
from typing import Dict, Any

# --- Import Core Components ---
from shared_libs.orchestrators.cv_pipeline_factory import CVPipelineFactory
from shared_libs.orchestrators.cv_training_orchestrator import CVTrainingOrchestrator
from shared_libs.core_utils.config_manager import ConfigManager # For loading the config file
from shared_libs.core_utils.exceptions import WorkflowExecutionError, ConfigurationError

# NOTE: Set up basic logging before running
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_training_job(config_path: str, run_id: str) -> None:
    """
    The main entry point for a CV training job. 
    
    Delegates all setup, validation, and execution to the CVPipelineFactory and Orchestrator.
    """
    logger.info(f"--- Starting MLOps Training Job: {run_id} ---")
    
    try:
        # 1. LOAD CONFIGURATION (Uses shared utility)
        raw_config = ConfigManager.load_config(config_path)
        
        # 2. CREATE ORCHESTRATOR VIA FACTORY (The critical DI step)
        # Factory will handle Pydantic validation and inject all MLOps services (Logger, Registry).
        orchestrator: CVTrainingOrchestrator = CVPipelineFactory.create(
            config=raw_config,
            orchestrator_id=run_id
        )
        
        # 3. EXECUTE THE WORKFLOW
        # The orchestrator handles the entire lifecycle (Data -> Train -> Eval -> Register)
        final_metrics, model_uri = orchestrator.run()
        
        logger.info(f"Job completed successfully. Final Model URI: {model_uri}")
        logger.info(f"Final Metrics: {final_metrics}")
        
    except (ConfigurationError, ImportError, WorkflowExecutionError) as e:
        logger.critical(f"FATAL: Training workflow failed for run {run_id}. Error: {e}", exc_info=True)
        sys.exit(1) # Exit with non-zero code for MLOps pipeline failure
    except Exception as e:
        logger.critical(f"UNHANDLED FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    # --- Example Operational Entry Point ---
    
    # In a real environment, these are passed via command line arguments or environment variables
    
    # Giả định đường dẫn cấu hình chuẩn
    DEFAULT_CONFIG_PATH = "./config/medical_training_config.yaml" 
    JOB_RUN_ID = os.environ.get("MLOPS_JOB_ID", "local_train_001")
    
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        logger.error(f"Config file not found at {DEFAULT_CONFIG_PATH}. Cannot run.")
        sys.exit(1)

    run_training_job(
        config_path=DEFAULT_CONFIG_PATH,
        run_id=JOB_RUN_ID
    )