# shared_libs/data_labeling/manual_annotation/factory.py

import logging
from typing import Dict, Any, Type
from .base_manual_annotator import BaseManualAnnotator

# Import các Parser chuyên biệt
from .classification_parser import ClassificationParser
from .detection_parser import DetectionParser
from .segmentation_parser import SegmentationParser
from .ocr_parser import OCRParser
# (EmbeddingParser ít phổ biến hơn cho manual, nhưng có thể thêm nếu cần)

logger = logging.getLogger(__name__)

class ManualAnnotatorFactory:
    """
    Factory class quản lý việc tạo ra các Manual Annotator (Parser) chuyên biệt.
    Nó quyết định Parser nào sẽ được sử dụng dựa trên task_type trong cấu hình.
    """
    
    # Ánh xạ task_type (domain) sang Parser Class
    MANUAL_ANNOTATOR_MAPPING: Dict[str, Type[BaseManualAnnotator]] = {
        "classification": ClassificationParser,
        "detection": DetectionParser,
        "segmentation": SegmentationParser,
        "ocr": OCRParser,
        # Thêm các parser khác nếu cần
    }

    @staticmethod
    def get_annotator(domain_type: str, config: Dict[str, Any]) -> BaseManualAnnotator:
        """
        Tạo và trả về một Manual Annotator (Parser) instance.

        Args:
            domain_type (str): Loại tác vụ CV ('classification', 'detection', ...).
            config (Dict[str, Any]): Cấu hình chi tiết cho Parser (ví dụ: format, columns).

        Returns:
            BaseManualAnnotator: Một instance của Parser class tương ứng.

        Raises:
            ValueError: Nếu domain_type không được hỗ trợ.
            RuntimeError: Nếu quá trình khởi tạo thất bại.
        """
        domain_type = domain_type.lower()
        if domain_type not in ManualAnnotatorFactory.MANUAL_ANNOTATOR_MAPPING:
            raise ValueError(f"Unsupported manual annotation domain: {domain_type}")

        AnnotatorClass = ManualAnnotatorFactory.MANNOTATOR_MAPPING[domain_type]
        
        try:
            # Khởi tạo Parser (nó sẽ tự validate config bằng Pydantic Schema của mình)
            return AnnotatorClass(config)
        except Exception as e:
            logger.error(f"Failed to instantiate {domain_type} Manual Annotator: {e}")
            raise RuntimeError(f"Manual Annotator creation failed: {e}")