# cv_factory/api_service/endpoints/prediction_router.py (HARDENED)

from fastapi import APIRouter, Body, HTTPException, Depends
import logging
import base64

# Import Core Orchestrator Contract
from shared_libs.orchestrators.cv_inference_orchestrator import CVInferenceOrchestrator 
from shared_libs.core_utils.exceptions import WorkflowExecutionError, DataIntegrityError
from cv_factory.api_service.schemas.service_schemas import PredictionInput, PredictionOutput

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency: Logic phải được cung cấp bởi tầng Setup/Factory
# Đây là nơi CVPipelineFactory inject Orchestrator đã cấu hình (Singleton)
def get_inference_orchestrator() -> CVInferenceOrchestrator:
    """
    Dependency function to retrieve the singleton instance of the fully configured CVInferenceOrchestrator.
    
    NOTE: In a real FastAPI app, this would retrieve the instance stored in the global app state.
    """
    # For demonstration, we assume a placeholder retrieval
    if 'orchestrator_instance' not in globals():
        logger.critical("Inference Orchestrator not found in application state.")
        raise HTTPException(status_code=503, detail="Inference service not initialized.")
    return globals()['orchestrator_instance'] # Placeholder for global instance retrieval

@router.post("/predict", response_model=PredictionOutput, summary="Run Internal CV Inference using the MLOps Orchestrator")
def run_inference(
    input_data: PredictionInput = Body(...),
    # Inject the Orchestrator instead of a simple HTTP client
    orchestrator: CVInferenceOrchestrator = Depends(get_inference_orchestrator) 
):
    """
    Accepts Base64 image data and delegates the full Preprocessing->Predict->Postprocessing 
    workflow to the internally managed CVInferenceOrchestrator.
    """
    
    try:
        # 1. Prepare Raw Input (Dictionary cho Adapter/Router)
        # CVPredictor.preprocess() sẽ nhận Dict này và decode Base64
        raw_input_dict = {
            "image_base64": input_data.image_base64,
            "threshold": input_data.threshold, # Tham số được truyền cho Postprocessor Domain
            "metadata": input_data.metadata
        }
        
        # 2. Delegation to Orchestrator (Orchestrator gọi Predictor.predict_pipeline)
        # Orchestrator.run() mong đợi một list các inputs (Batch/API).
        final_predictions_list = orchestrator.run(inputs=[raw_input_dict])
        
        # 3. Validation và Conversion to Output Schema
        if not final_predictions_list:
            raise WorkflowExecutionError("Orchestrator returned no predictions.")

        # Giả sử output của MedicalPostprocessor (FinalDiagnosis Entity) đã tuân thủ PredictionOutput
        # Chúng ta chỉ lấy phần tử đầu tiên trong batch trả về
        return PredictionOutput(**final_predictions_list[0]) 
        
    except DataIntegrityError as e:
        logger.warning(f"Input validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Input data integrity error: {e}")
    except WorkflowExecutionError as e:
        # Lỗi được bắt từ tầng Orchestration (MLOps error, model load fail, etc.)
        logger.error(f"Inference workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Workflow Execution Error: {e}")
    except Exception as e:
        logger.critical(f"Unhandled Prediction Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction service error.")