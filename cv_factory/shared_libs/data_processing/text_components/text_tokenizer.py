# shared_libs/data_processing/text_components/text_tokenizer.py
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from .base_text_processor import BaseTextProcessor, RawTextData, TokenizedData

logger = logging.getLogger(__name__)

class TextTokenizer(BaseTextProcessor):
    """
    Component chuyển đổi Raw Text thành Token ID (integer) và áp dụng Padding/Truncation.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_length = self.config.get("max_length", 512)
        self.vocab_size = self.config.get("vocab_size", 10000)
        self.padding_strategy = self.config.get("padding_strategy", "post")
        logger.info(f"TextTokenizer initialized. Max length: {self.max_length}")

    def _mock_tokenize(self, text: str) -> List[int]:
        """Mô phỏng Tokenization đơn giản (word index)."""
        # Giả lập 10000 từ vựng
        words = text.lower().split()
        tokens = [hash(w) % self.vocab_size for w in words]
        return tokens

    def process(self, text_data: RawTextData) -> TokenizedData:
        """
        Tokenize và Padding/Truncate.
        """
        if isinstance(text_data, str):
            token_list = [self._mock_tokenize(text_data)]
        else:
            token_list = [self._mock_tokenize(t) for t in text_data]
            
        final_tokens = []
        for tokens in token_list:
            # 1. Truncate
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                
            # 2. Padding
            padding_needed = self.max_length - len(tokens)
            if padding_needed > 0:
                padding = [0] * padding_needed
                if self.padding_strategy == "post":
                    tokens.extend(padding)
                else:
                    tokens = padding + tokens
            
            final_tokens.append(np.array(tokens, dtype=np.int32))
            
        # Nếu chỉ có 1 chuỗi, trả về mảng 1D
        if len(final_tokens) == 1 and isinstance(text_data, str):
            return final_tokens[0]
            
        logger.debug(f"Text tokenized and padded to shape: {final_tokens[0].shape if final_tokens else 'Empty'}.")
        
        return np.array(final_tokens)