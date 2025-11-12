# shared_libs/data_processing/text_components/text_loader.py
import logging
from typing import Dict, Any, Optional
from .base_text_processor import BaseTextProcessor, RawTextData, TokenizedData

logger = logging.getLogger(__name__)

class TextLoader(BaseTextProcessor):
    """
    Component tải dữ liệu Text thô từ file annotation hoặc file mô tả.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.encoding = self.config.get("encoding", "utf-8")
        self.output_format = self.config.get("output_format", "single_string")
        logger.info(f"TextLoader initialized. Encoding: {self.encoding}")

    def _mock_load(self, path: str) -> str:
        """Mô phỏng quá trình tải Text."""
        return "The quick brown fox jumps over the lazy dog. Text data for OCR processing."

    def process(self, text_path: str) -> RawTextData:
        """
        Tải Text từ đường dẫn.
        """
        raw_text = self._mock_load(text_path)
        
        # HARDENING: Kiểm tra và làm sạch ký tự cơ bản
        if self.config.get('clean_unicode', True):
            raw_text = raw_text.encode("ascii", "ignore").decode("ascii")

        if self.output_format == "list_of_sentences":
            # Tách thành list of strings (mô phỏng)
            return raw_text.split(". ")
            
        logger.debug(f"Loaded raw text data (length: {len(raw_text)}).")
        
        return raw_text