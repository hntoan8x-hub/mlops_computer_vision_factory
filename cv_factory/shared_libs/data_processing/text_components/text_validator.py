# shared_libs/data_processing/text_components/text_validator.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from .base_text_processor import BaseTextProcessor, RawTextData, TokenizedData

logger = logging.getLogger(__name__)

class TextValidator(BaseTextProcessor):
    """
    Component xác thực Text: kiểm tra độ dài tối thiểu, tỷ lệ từ ngoài từ điển.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_length = self.config.get("min_length", 5) 
        self.max_length_tokens = self.config.get("max_length_tokens", 512)
        logger.info(f"TextValidator initialized. Min Length: {self.min_length}")

    def process(self, text_data: RawTextData) -> TokenizedData:
        """
        Kiểm tra và trả về dữ liệu token (Giả định token đã được tạo).
        """
        # Nếu input là TokenizedData (đã qua Tokenizer)
        if isinstance(text_data, np.ndarray):
            # 1. Kiểm tra độ dài token
            if text_data.ndim > 1:
                # Nếu là batch/list of tokens
                for tokens in text_data:
                    if len(tokens) > self.max_length_tokens:
                        logger.warning(f"Token sequence exceeds configured max length: {len(tokens)} > {self.max_length_tokens}.")
            
            # Trả về token đã được validate
            return text_data
            
        # Nếu input là RawTextData (chưa qua Tokenizer, cần logic validate string)
        elif isinstance(text_data, str):
            if len(text_data.split()) < self.min_length:
                 raise ValueError(f"Text too short ({len(text_data.split())} words). Minimum required: {self.min_length}")
                 
            # Vì BaseTextProcessor.process phải trả về TokenizedData, 
            # nếu TextValidator nhận raw text, nó cần phải tokenization hoặc raise error
            raise TypeError("TextValidator expects TokenizedData (Numpy array) as input after Tokenizer.")

        # Trả về input (nếu không có gì thay đổi)
        return text_data