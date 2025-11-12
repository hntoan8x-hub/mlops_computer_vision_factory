# shared_libs/data_processing/text_components/base_text_processor.py
import abc
from typing import Any, Dict, Optional, Union, List
import numpy as np

# Định nghĩa các kiểu dữ liệu cho Text
RawTextData = Union[str, List[str]] # Text thô (string hoặc list of strings)
TokenizedData = Union[np.ndarray, List[np.ndarray]] # Dữ liệu đã tokenize (List of integers/Numpy array)

class BaseTextProcessor(abc.ABC):
    """
    Abstract Base Class cho tất cả các Text Processor.

    Định nghĩa giao diện chuẩn để xử lý dữ liệu ngôn ngữ (text) trong ngữ cảnh CV (OCR, VQA).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Khởi tạo với cấu hình."""
        self.config = config if config is not None else {}

    @abc.abstractmethod
    def process(self, text_data: RawTextData) -> TokenizedData:
        """
        Áp dụng quá trình xử lý cho dữ liệu Text.

        Args:
            text_data (RawTextData): Dữ liệu text thô đầu vào.

        Returns:
            TokenizedData: Dữ liệu đã được chuyển đổi (thường là mảng token ID đã padding).
        
        Raises:
            NotImplementedError: Nếu phương thức chưa được triển khai.
        """
        raise NotImplementedError