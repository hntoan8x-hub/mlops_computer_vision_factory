# api_service/sads_router.py (UPDATED - ADDING HEALTH CHECK ENDPOINTS)

from fastapi import APIRouter, Body, Depends, HTTPException, status
from typing import Dict, Any, Union, Optional
from datetime import datetime

# --- Import Service và Dependencies ---
from domain_models.surface_anomaly_detection.sads_inference_service import SADSInferenceService
from .dependencies import get_sads_service

router = APIRouter(
    prefix="/sads/v1",
    tags=["Surface Anomaly Detection System (SADS)"]
)

# Thêm Router cho Health Check (Không sử dụng prefix /sads/v1/ mà sử dụng /health)
health_router = APIRouter(
    prefix="/health",
    tags=["Health Check"]
)

# Request Body Schema (Đảm bảo input là JSON chuẩn)
class PredictionPayload(Dict):
    image_base64: str = Body(..., description="Image data encoded in Base64 format.")
    metadata: Optional[Dict[str, Any]] = None

# --- HEALTH CHECK ENDPOINTS (NEW) ---

@health_router.get("/liveness")
async def liveness_check():
    """Kiểm tra Liveness: Ứng dụng có đang chạy không."""
    return {"status": "UP", "timestamp": datetime.now().isoformat()}

@health_router.get("/readiness")
async def readiness_check(sads_service: SADSInferenceService = Depends(get_sads_service)):
    """
    Kiểm tra Readiness: Ứng dụng có sẵn sàng nhận traffic không.
    Sử dụng Dependency Injection để kiểm tra nếu Service đã được khởi tạo.
    """
    # Nếu get_sads_service() không raise exception, service đã được khởi tạo
    return {"status": "READY", "timestamp": datetime.now().isoformat()}

@health_router.get("/model_readiness")
async def model_readiness_check(sads_service: SADSInferenceService = Depends(get_sads_service)):
    """
    [HARDENED] Kiểm tra Model Readiness: Đảm bảo TẤT CẢ các mô hình SADS đã được tải.
    """
    try:
        # Kiểm tra trạng thái tải mô hình trong SADS Pipeline
        # Giả định SADSInferenceService.pipeline.orchestrator có thuộc tính check_model_loaded()
        
        # MOCKING: Giả định Orchestrator SADS có 3 Predictor:
        if not sads_service.pipeline.orchestrator.detection_predictor.is_loaded:
            raise RuntimeError("Detection model not loaded.")
        if not sads_service.pipeline.orchestrator.classification_predictor.is_loaded:
            raise RuntimeError("Classification model not loaded.")
        if not sads_service.pipeline.orchestrator.segmentation_predictor.is_loaded:
            raise RuntimeError("Segmentation model not loaded.")
            
        return {"status": "MODELS_LOADED", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model loading check failed: {e}"
        )


# --- SADS PREDICT ENDPOINT (Giữ nguyên logic chính) ---

@router.post("/predict")
async def predict_sads(
    payload: Dict[str, Any] = Body(..., example={"image_base64": "iVBORw0K...", "metadata": {"unit_id": "A10"}}),
    sads_service: SADSInferenceService = Depends(get_sads_service) 
):
    """
    Thực thi SADS Inference Pipeline (Detection -> Classification -> Segmentation -> Decision Engine).
    """
    
    # ... (Logic validation và ủy quyền giữ nguyên) ...
    if 'image_base64' not in payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'image_base64' field in payload."
        )

    try:
        result = sads_service.predict(raw_input=payload)
        
        return result
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SADS Execution Error: {e}"
        )