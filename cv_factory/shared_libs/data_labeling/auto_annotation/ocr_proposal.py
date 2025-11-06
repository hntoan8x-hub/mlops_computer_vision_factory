# shared_libs/data_labeling/auto_annotation/ocr_proposal.py

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from torch import Tensor
import os

from .base_auto_annotator import BaseAutoAnnotator, StandardLabel
from ...configs.label_schema import OCRLabel, OCRToken

logger = logging.getLogger(__name__)

class OCRProposalAnnotator(BaseAutoAnnotator):
    """
    Annotator chuyên biệt cho OCR/Text Extraction: Phát hiện vùng text (bbox) 
    và trích xuất nội dung (full_text, tokens).
    """
    
    def _run_inference(self, image_data: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Mô phỏng chạy mô hình OCR (ví dụ: EasyOCR, PaddleOCR) và trả về danh sách: 
        [(bbox_xyxy_pixel, text_content, confidence), ...]
        
        Args:
            image_data (np.ndarray): Ảnh đầu vào (H, W, C).

        Returns:
            List[Tuple[Tuple[int, int, int, int], str, float]]: Danh sách các dự đoán (BBox, Nội dung, Độ tin cậy).
        """
        H, W, _ = image_data.shape
        
        # Giả lập kết quả dự đoán của một dòng văn bản
        predictions = [
            # Word 1: Hello
            ((int(W*0.1), int(H*0.1), int(W*0.3), int(H*0.2)), "Hello", 0.99),
            # Word 2: World!
            ((int(W*0.3), int(H*0.1), int(W*0.5), int(H*0.2)), "World!", 0.95),
            # Text có độ tin cậy thấp (sẽ bị loại bỏ)
            ((int(W*0.7), int(H*0.7), int(W*0.8), int(H*0.8)), "noise", 0.60),
        ]
        return predictions

    def _normalize_output(self, raw_prediction: List[Tuple[Tuple, str, float]], metadata: Dict[str, Any]) -> List[StandardLabel]:
        """
        Chuẩn hóa kết quả dự đoán thô thành đối tượng OCRLabel (Pydantic), bao gồm 
        full_text (chuỗi đầy đủ) và tokens (chi tiết từng từ/ký tự).
        """
        image_path: str = metadata.get("image_path", "unknown")
        
        # Giả định kích thước ảnh có sẵn từ ảnh thô
        img_h, img_w, _ = metadata.get("image_data").shape 
        
        full_text_list: List[str] = []
        suggested_tokens: List[OCRToken] = []

        for bbox_raw, text_content, confidence in raw_prediction:
            if confidence >= self.min_confidence:
                
                # 1. Chuẩn hóa Bounding Box về [0, 1]
                x_min, y_min, x_max, y_max = bbox_raw
                bbox_normalized: Tuple[float, float, float, float] = (
                    x_min / img_w,
                    y_min / img_h,
                    x_max / img_w,
                    y_max / img_h
                )
                
                # 2. Tạo đối tượng token đã được kiểm tra (Pydantic)
                try:
                    token = OCRToken(
                        text=text_content, 
                        bbox=bbox_normalized
                    )
                    suggested_tokens.append(token)
                    full_text_list.append(text_content)
                except Exception as e:
                    logger.warning(f"Invalid OCR token created: {e}")

        # 3. Tổng hợp Full Text (nối các token lại)
        full_text = " ".join(full_text_list)
        
        # 4. Tạo đối tượng OCRLabel cuối cùng
        if suggested_tokens:
            ocr_label = OCRLabel(
                image_path=image_path,
                full_text=full_text,
                tokens=suggested_tokens
            )
            # NOTE: Phải trả về một List[StandardLabel] để nhất quán với BaseAnnotator
            return [ocr_label]
        else:
            return []