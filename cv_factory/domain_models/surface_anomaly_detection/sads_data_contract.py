# domain_models/surface_anomaly_detection/sads_data_contract.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# --- 1. Entity Lỗi Domain Thống nhất ---
@dataclass
class SADSDefect:
    """Thực thể lỗi bề mặt thống nhất sau khi hợp nhất 3 outputs."""
    bbox: List[float]               # Bounding Box chuẩn hóa [xmin, ymin, xmax, ymax]
    cls_id: int                     # ID lớp lỗi (từ Classification)
    score: float                    # Điểm tổng hợp (ví dụ: trung bình Det và Cls)
    mask: Optional[np.ndarray]      # Binary mask của lỗi (từ Segmentation)
    area_px: Optional[float] = None # Diện tích lỗi (pixel)
    
# --- 2. Hợp đồng Kết quả cuối cùng ---
@dataclass
class SADSFinalResult:
    """Hợp đồng đầu ra cuối cùng sau Decision Engine."""
    final_decision: str             # PASS / FAIL / ESCALATE
    details: List[tuple[str, SADSDefect]] # Chi tiết quyết định cho từng lỗi