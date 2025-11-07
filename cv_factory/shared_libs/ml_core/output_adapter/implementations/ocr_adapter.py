# shared_libs/ml_core/output_adapter/implementations/ocr_adapter.py

import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple
import logging

from ..base_output_adapter import BaseOutputAdapter, RawModelOutput, StandardizedOutput

from ...output_adapter.configs.output_adapter_config_schema import OutputAdapterConfig, OCRAdapterParams

logger = logging.getLogger(__name__)

class OCRAdapter(BaseOutputAdapter):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adapter_config = OutputAdapterConfig(**config)
        self.params: OCRAdapterParams = self.adapter_config.params 
        # Giả định: Tải TOKEN_TO_CHAR_MAP từ config.tokenizer_config
        self.TOKEN_TO_CHAR_MAP = {1: 'h', 2: 'e', 3: 'l', 4: 'o', 5: ' '} # Ví dụ

    def adapt(self, raw_output: RawModelOutput, **kwargs: Dict[str, Any]) -> StandardizedOutput:
        """
        Xử lý đầu ra thô của mô hình OCR.

        Args:
            raw_output (Dict): Đầu ra thô của mô hình. Giả định:
                               {'token_indices': Tensor/np.ndarray [N], 
                                'bboxes': Tensor/np.ndarray [N, 4], 
                                'confidences': Tensor/np.ndarray [N]}
            **kwargs: Metadata (ví dụ: 'tokenizer_config' chứa VOCAB).

        Returns:
            List[Dict]: Danh sách các dự đoán chuẩn hóa: 
                        [{'text': str, 'bbox': np.ndarray, 'confidence': float}, ...]
        """
        
        # 1. Chuyển Tensors về NumPy
        token_indices = self._to_numpy(raw_output.get('token_indices'))
        bboxes = self._to_numpy(raw_output.get('bboxes'))
        confidences = self._to_numpy(raw_output.get('confidences'))
        
        if token_indices is None or bboxes is None or confidences is None:
             raise ValueError("OCR raw output must contain token_indices, bboxes, and confidences.")

        # 2. Xử lý Tokenization ngược (Token IDs -> Text)
        def decode_tokens(indices: np.ndarray) -> str:
            # Sử dụng self.TOKEN_TO_CHAR_MAP
            return "".join([self.TOKEN_TO_CHAR_MAP.get(idx, '<?>') for idx in indices.flatten() if idx != 0])
        
        # 3. Chuẩn hóa thành định dạng List[Dict] (giữ nguyên)
        predictions_list: List[Dict[str, Any]] = []
        
        # Giả định: Mỗi entry trong bboxes/confidences/token_indices tương ứng với một từ/vùng
        for bbox, confidence, token_idx in zip(bboxes, confidences, token_indices):
            # Giả định token_idx là ID của ký tự đầu tiên/word ID, hoặc cần logic phức tạp hơn
            
            # Ở đây, ta giả định token_idx là mảng ID cho toàn bộ văn bản trong bbox đó
            text_content = decode_tokens(token_idx) if token_idx.ndim > 0 else "text_placeholder"
            
            predictions_list.append({
                'text': text_content,
                'bbox': bbox,             # np.ndarray (xyxy format)
                'confidence': float(confidence)
            })

        return predictions_list