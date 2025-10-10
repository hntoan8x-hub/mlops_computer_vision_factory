# cv_factory/scripts/inference_main.py

import logging
import os
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any

# --- Import Core Components ---
from shared_libs.orchestrators.cv_pipeline_factory import CVPipelineFactory
from shared_libs.orchestrators.cv_inference_orchestrator import CVInferenceOrchestrator
from shared_libs.core_utils.config_manager import ConfigManager
from shared_libs.core_utils.exceptions import ConfigurationError

# Import API Router and Config (assumed location)
from cv_factory.api_service.endpoints.prediction_router import router as prediction_router
# NOTE: We'll reuse API_CONFIG logic or rely on environment variables

logger = logging.getLogger(__name__)

# Global Application State to hold the Singleton Orchestrator instance
app = FastAPI(
    title="CV MLOps Inference Service",
    version="1.0.0",
)

# 1. Initialize Global Application Dependencies (Will run before Uvicorn starts)
def initialize_dependencies(config_path: str):
    """
    Initializes the MLOps execution core and injects the Orchestrator into the FastAPI state.
    """
    logger.info("--- Initializing MLOps Inference Dependencies ---")
    
    try:
        # a. Load Configuration
        raw_config = ConfigManager.load_config(config_path)
        
        # b. Create Orchestrator via Factory (DI step)
        orchestrator: CVInferenceOrchestrator = CVPipelineFactory.create(
            config=raw_config,
            orchestrator_id="production-inference-api"
        )
        
        # c. INJECT INTO APPLICATION STATE (Critical for FastAPI/Depends mechanism)
        # This makes the Orchestrator available to prediction_router.py
        app.state.inference_orchestrator = orchestrator
        
        # d. Register Router after successful initialization
        app.include_router(prediction_router)
        
        logger.info("Dependencies injected. API Router configured.")
        
        # NOTE: Logic to ensure get_injected_orchestrator in prediction_router.py 
        # must be updated to retrieve from app.state.inference_orchestrator

    except ConfigurationError as e:
        logger.critical(f"FATAL: Configuration error during API initialization: {e}")
        raise
    except Exception as e:
        logger.critical(f"FATAL: Unhandled error during dependency initialization: {e}")
        raise


if __name__ == '__main__':
    # --- Example Operational Entry Point ---
    
    API_CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config/medical_inference_config.yaml")
    
    if not os.path.exists(API_CONFIG_PATH):
        logger.error(f"Config file not found at {API_CONFIG_PATH}. Cannot start API.")
        sys.exit(1)

    # Khởi tạo các dependency
    initialize_dependencies(config_path=API_CONFIG_PATH)
    
    # Lấy tham số vận hành từ config
    # NOTE: Trong thực tế, bạn sẽ load API_CONFIG để lấy host/port
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting FastAPI server on http://{HOST}:{PORT}")
    
    # Khởi chạy Uvicorn server
    uvicorn.run(
        "inference_main:app", 
        host=HOST, 
        port=PORT, 
        log_level="info", 
        reload=False # IMPORTANT: reload=False for production
    )