# shared_libs/data_labeling/implementations/ocr_labeler.py (Cập nhật)

import logging
from typing import Dict, Any, List, Union, Tuple, Literal
from torch import tensor, long
import pandas as pd

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import OCRLabel
from ...data_labeling.configs.labeler_config_schema import OCRLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class OCRLabeler(BaseLabeler):
    """
    Concrete Labeler cho OCR/Text Extraction. Hỗ trợ Manual Parsing (tải JSON) 
    và Auto Proposal (sinh text/bbox).
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.annotation_mode: Literal["manual", "auto"] = self.validated_config.get("annotation_mode", "manual")
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.model_dump().get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "ocr") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """Tải dữ liệu nhãn hoặc metadata ảnh tùy theo chế độ Annotation."""
        source_uri = self.validated_config.params.label_source_uri
        
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # CHẾ ĐỘ MANUAL: Parsing file nhãn (thường là JSON/XML)
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="ocr",
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[OCRLabel] = parser.parse(raw_data)
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"OCR manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            # CHẾ ĐỘ AUTO: Raw data là List[Dict] metadata ảnh
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Kiểm tra nhãn OCR phải có full_text và tokens."""
        return "full_text" in sample and "tokens" in sample

    def convert_to_tensor(self, label_data: Dict[str, Any]):
        """Chuyển đổi text thành tensor ID (đã padding) và BBox tensor."""
        # NOTE: Logic tokenization và padding cần được thực hiện ở đây.
        # Để đơn giản, giả định OCRLabelerConfig đã chứa thông tin cần thiết.
        
        # Mô phỏng việc tokenization/conversion
        text_tensor = tensor([1, 2, 3]).long() # Ví dụ: [token_id1, token_id2, ...]
        bbox_tensor = tensor([[0.1, 0.1, 0.5, 0.5]]).float() # Ví dụ: [[x1, y1, x2, y2]]
        
        return text_tensor, bbox_tensor