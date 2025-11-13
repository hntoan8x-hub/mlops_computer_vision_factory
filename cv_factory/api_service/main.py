# api_service/main.py (UPDATED - INCLUDING HEALTH ROUTER)

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import json
from typing import Dict, Any
import logging

from .sads_router import router, health_router # <<< NEW IMPORT: health_router >>>
from .dependencies import initialize_sads_service, set_app_config

logger = logging.getLogger("API_MAIN")

# --- MOCK CẤU HÌNH (Trong thực tế sẽ tải từ file/Vault) ---
try:
    with open('configs/sads_config.json', 'r') as f:
        APP_CONFIG = json.load(f)
except FileNotFoundError:
    logger.warning("Mock config file not found. Using Placeholder Config.")
    APP_CONFIG = {
        "models": {"detection": {"uri": "models:/det/latest"}, "classification": {"uri": "models:/cls/latest"}, "segmentation": {"uri": "models:/seg/latest"}},
        "domain": {"postprocessor": {"params": {"defect_confidence_threshold": 0.70, "max_area_cm2": 0.5, "max_allowed_defects": 1}}},
        "device": "cpu"
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng FastAPI.
    Logic Khởi tạo (Startup) nằm ở đây.
    """
    logger.info("--- Application Startup: Setting Config & Initializing Services ---")
    
    # 1. Tiêm cấu hình vào Dependency Manager
    set_app_config(APP_CONFIG)
    
    # 2. KHỞI TẠO SERVICE: Tải tất cả các mô hình nặng (chỉ 1 lần)
    initialize_sads_service()
    
    yield # Ứng dụng chạy
    
    # Logic Dọn dẹp (Shutdown) nằm ở đây
    logger.info("--- Application Shutdown ---")


app = FastAPI(
    title="AI Factory - SADS Production API",
    version="1.0.0",
    description="Endpoint cho Hệ thống Phát hiện Bất thường Bề mặt (SADS) đa mô hình.",
    lifespan=lifespan
)

# Thêm Routers
app.include_router(router)
app.include_router(health_router) # <<< NEW: INCLUDE HEALTH CHECK ROUTER >>>

if __name__ == "__main__":
    # Dùng để chạy local: python api_service/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)