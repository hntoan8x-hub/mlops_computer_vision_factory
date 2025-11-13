# domain_models/surface_anomaly_detection/sads_postprocessor.py (HARDENED)

import logging
from typing import Dict, Any, List, Union, Tuple
from .sads_config_schema import SADSPostprocessorParams # Import tham số nghiệp vụ
from .sads_data_contract import SADSDefect, SADSFinalResult # Import Data Contract mới
import numpy as np
from dataclasses import asdict

logger = logging.getLogger(__name__)

# NOTE: Giả định utility NMS có sẵn
# Không áp dụng NMS ở đây nữa; nó nên được áp dụng ở Detection Predictor hoặc đầu SADS Orchestrator.
# Postprocessor chỉ tập trung vào Business Rules (luật kinh doanh).

class SADSPostprocessor:
    """
    Lớp Postprocessor Domain-specific, hoạt động như Decision Engine.
    Áp dụng các Luật Kinh Doanh (Business Rules) lên các SADSDefect Entity.
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Khởi tạo với tham số nghiệp vụ đã được validate (từ SADSPostprocessorParams).
        """
        try:
            self.params = SADSPostprocessorParams(**kwargs)
        except Exception as e:
            raise ValueError(f"SADS Postprocessor Config validation failed: {e}")

        logger.info(f"SADS Postprocessor initialized. Max Allowed Defects: {self.params.max_allowed_defects}")

    def _get_defect_area_cm2(self, defect: SADSDefect) -> float:
        """
        Mô phỏng chuyển đổi diện tích pixel sang cm².
        Trong thực tế, cần một tỷ lệ Px/Cm (từ camera calibration).
        """
        # Giả định tỷ lệ cố định để mô phỏng
        PIXEL_TO_CM2_RATIO = 0.0001
        if defect.area_px is not None:
            return defect.area_px * PIXEL_TO_CM2_RATIO
        return 0.0

    def _make_item_decision(self, defect: SADSDefect) -> str:
        """
        Quyết định (Decision) cho MỘT ứng viên lỗi (Item-level Decision).
        Thực hiện các Business Rules Real-World.
        """
        decision = "PASS"
        area_cm2 = self._get_defect_area_cm2(defect)

        # Rule 1: Confidence quá thấp → ESCALATE (Yêu cầu kiểm tra thủ công)
        if defect.score < self.params.defect_confidence_threshold:
            decision = "ESCALATE"
            return decision

        # Rule 2: Diện tích lỗi vượt quá ngưỡng cho phép (Sử dụng đơn vị cm²)
        if self.params.max_area_cm2 > 0 and area_cm2 > self.params.max_area_cm2:
            decision = "FAIL"
            return decision

        # Rule 3: Lỗi thuộc danh sách lỗi nghiêm trọng (Critical Defects)
        # Giả định cần một Mapping từ cls_id sang tên lớp (không có ở đây, dùng ID mô phỏng)
        CRITICAL_DEFECT_IDS = [1, 5, 8] # Ví dụ: Lõm sâu (1), Vết nứt (5), Cháy (8)
        if defect.cls_id in CRITICAL_DEFECT_IDS:
            decision = "FAIL"
            return decision
            
        return decision

    def decide(self, unified_defects: List[SADSDefect]) -> SADSFinalResult:
        """
        [NEW SIGNATURE] Thực thi Decision Engine trên các SADSDefect Entity.
        Trả về Quyết định cuối cùng (Frame-level Decision).
        """
        item_results: List[Tuple[str, SADSDefect]] = []
        
        # 1. Quyết định từng Item (Item-level decision)
        for defect in unified_defects:
            item_decision = self._make_item_decision(defect)
            item_results.append((item_decision, defect))

        # 2. Áp dụng Rule Tổng Thể (Frame-level decision)
        final_decision = "PASS"
        
        # Rule Tổng Thể 1: Nếu CÓ BẤT KỲ lỗi nào FAIL → Frame FAIL
        if any(r[0] == "FAIL" for r in item_results):
            final_decision = "FAIL"
        
        # Rule Tổng Thể 2: Nếu không FAIL, nhưng có BẤT KỲ lỗi nào ESCALATE → Frame ESCALATE
        elif any(r[0] == "ESCALATE" for r in item_results):
            final_decision = "ESCALATE"
            
        # Rule Tổng Thể 3: Số lượng lỗi tổng thể vượt quá giới hạn (ngay cả khi chúng là PASS)
        if len(item_results) > self.params.max_allowed_defects and final_decision == "PASS":
            final_decision = "FAIL" # Có quá nhiều lỗi nhỏ tích lũy

        logger.info(f"Final Frame Decision: {final_decision}. Total Defects: {len(unified_defects)}")

        # 3. Trả về Hợp đồng đầu ra
        return SADSFinalResult(
            final_decision=final_decision,
            details=item_results
        )