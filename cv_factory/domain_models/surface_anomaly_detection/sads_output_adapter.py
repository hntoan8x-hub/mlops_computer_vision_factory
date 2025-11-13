# domain_models/surface_anomaly_detection/sads_output_adapter.py

from typing import List, Dict, Any, Tuple
import numpy as np
from .sads_data_contract import SADSDefect # Import Entity

class SADSOutputAdapter:
    """
    Adapter Domain-specific: Hợp nhất outputs chuẩn hóa từ 3 mô hình 
    (Detection, Classification, Segmentation) thành entity SADSDefect.
    """
    def __init__(self, **kwargs):
        # Khởi tạo tham số adapter (nếu có)
        pass
        
    def assemble(self, 
                 det_output: Dict[str, Any], 
                 cls_result: Dict[str, Any], 
                 seg_mask: np.ndarray
                ) -> SADSDefect:
        """Thực hiện việc hợp nhất cho một ứng viên lỗi duy nhất."""
        
        # 1. Tính toán Diện tích
        area_px = float(np.sum(seg_mask))
        
        # 2. Tính toán Score Hợp nhất
        # Giả định Prediction Output (Detection Adapter) có key 'score'
        # Giả định Classification Output (Classification Adapter) có key 'score'
        # Giả định Output Adapter chuẩn hóa trả về 'class' (int)
        combined_score = (det_output['score'] + cls_result['score']) / 2.0
        
        # 3. Tạo Entity
        return SADSDefect(
            bbox=det_output['box'].tolist(), 
            cls_id=int(cls_result['class']),
            score=combined_score,
            mask=seg_mask,
            area_px=area_px
        )

    def assemble_batch(self, batch_data: List[Tuple[Dict, Dict, np.ndarray]]) -> List[SADSDefect]:
        """Hợp nhất một lô (batch) các ứng viên lỗi."""
        defects = []
        for det, cls, seg in batch_data:
            try:
                defects.append(self.assemble(det, cls, seg))
            except Exception as e:
                print(f"Lỗi hợp nhất ứng viên: {e}") 
                continue
        return defects