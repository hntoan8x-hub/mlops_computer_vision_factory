# shared_libs/data_processing/text_components/text_augmenter.py
import logging
import random
from typing import Dict, Any, Optional
from .base_text_processor import BaseTextProcessor, RawTextData, TokenizedData

logger = logging.getLogger(__name__)

class TextAugmenter(BaseTextProcessor):
    """
    Component áp dụng các kỹ thuật Augmentation cho Text (ví dụ: Thay thế từ đồng nghĩa, lỗi chính tả).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.apply_typo = self.config.get("apply_typo", False)
        self.swap_rate = self.config.get("swap_rate", 0.1)
        logger.info(f"TextAugmenter initialized. Apply Typo: {self.apply_typo}")

    def _mock_apply_typo(self, text: str) -> str:
        """Mô phỏng thêm lỗi chính tả/hoán đổi ký tự."""
        if len(text) < 5 or random.random() > 0.5:
            return text
            
        idx = random.randint(1, len(text) - 2)
        # Hoán đổi 2 ký tự liên tiếp
        text_list = list(text)
        text_list[idx], text_list[idx+1] = text_list[idx+1], text_list[idx]
        return "".join(text_list)
        
    def process(self, text_data: RawTextData) -> RawTextData:
        
        if not self.config.get('enabled', False):
            return text_data
            
        current_text = text_data
        
        if isinstance(current_text, str):
            if self.apply_typo:
                current_text = self._mock_apply_typo(current_text)
                logger.debug("Applied text typo/swap augmentation.")
        
        # NOTE: Augmentation thường chỉ áp dụng cho RawTextData. 
        # Nếu input là TokenizedData (do đã qua Tokenizer), cần phải reverse hoặc cấu trúc lại.
        # Ở đây, giả định nó được gọi trước Tokenizer, nên trả về RawTextData.

        return current_text