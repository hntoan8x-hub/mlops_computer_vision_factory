# shared_libs/data_labeling/manual_annotation/ocr_parser.py

import logging
from typing import List, Dict, Any, Union
import json
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import OCRLabel, OCRToken
from ....data_labeling.configs.labeler_config_schema import OCRLabelerConfig

logger = logging.getLogger(__name__)

class OCRParser(BaseManualAnnotator):
    """
    Parser chuyên biệt cho OCR: Xử lý file JSON/XML chứa full text, token và Bounding Box.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.parser_config = OCRLabelerConfig(**config) 
        
    def parse(self, raw_input: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[StandardLabel]:
        """
        Phân tích dữ liệu thô (thường là List of Dicts từ JSON) và tạo OCRLabel objects.

        Args:
            raw_input (List[Dict] | Dict): Dữ liệu nhãn thô (thường là danh sách các document/ảnh).
        """
        if isinstance(raw_input, dict):
            # Nếu input là nội dung JSON lớn, chuyển sang list để xử lý
            raw_list = [raw_input]
        elif isinstance(raw_input, list):
            raw_list = raw_input
        else:
            raise TypeError("OCRParser expects a List[Dict] or Dict input.")
            
        annotated_labels: List[OCRLabel] = []
        
        for item in raw_list:
            image_path = item.get("image_path")
            full_text = item.get("full_text")
            tokens_data = item.get("tokens", []) # List[Dict] của token/bbox
            
            if not image_path or not full_text:
                logger.warning(f"Skipping OCR entry due to missing image_path or full_text.")
                continue

            # 1. Chuyển đổi tokens thô sang Pydantic OCRToken
            validated_tokens: List[OCRToken] = []
            for token_data in tokens_data:
                try:
                    # Giả định token_data là {'text': str, 'bbox': [x,y,w,h] or (x,y,x,y)}
                    validated_tokens.append(OCRToken(**token_data))
                except Exception as e:
                    logger.warning(f"Invalid OCR Token detected for {image_path}: {e}")
            
            # 2. Tạo đối tượng OCRLabel tổng thể
            try:
                validated_label = OCRLabel(
                    image_path=image_path,
                    full_text=full_text,
                    tokens=validated_tokens
                )
                annotated_labels.append(validated_label)
            except Exception as e:
                logger.error(f"Failed to create valid OCRLabel for {image_path}: {e}")
                
        return annotated_labels