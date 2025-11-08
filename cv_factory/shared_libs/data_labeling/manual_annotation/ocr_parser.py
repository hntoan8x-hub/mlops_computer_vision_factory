# shared_libs/data_labeling/manual_annotation/ocr_parser.py (Hardened)

import logging
from typing import List, Dict, Any, Union
import json
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import OCRLabel, OCRToken
from ....data_labeling.configs.labeler_config_schema import OCRLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class OCRParser(BaseManualAnnotator):
    """
    Specialized Parser for OCR labels: Handles JSON/XML files containing full text, tokens, and Bounding Boxes.
    
    This parser ensures individual tokens and the overall structure comply with OCRLabel Pydantic schema.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the parser and validates its configuration.
        """
        super().__init__(config)
        try:
            self.parser_config = OCRLabelerConfig(**config) 
        except ValidationError as e:
            logger.critical(f"OCRParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")
        
    def parse(self, raw_input: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[StandardLabel]:
        """
        Parses raw data (typically a list of Dicts from JSON) and creates validated OCRLabel objects.

        Args:
            raw_input: The raw label data (List[Dict] for multiple documents/images, or Dict for one).
        
        Returns:
            List[StandardLabel]: List of validated OCRLabel objects.

        Raises:
            TypeError: If raw_input type is unsupported.
        """
        if isinstance(raw_input, dict):
            raw_list = [raw_input]
        elif isinstance(raw_input, list):
            raw_list = raw_input
        else:
            raise TypeError("OCRParser expects a List[Dict] or Dict input.")
            
        annotated_labels: List[OCRLabel] = []
        
        for item in raw_list:
            image_path = item.get("image_path")
            full_text = item.get("full_text")
            tokens_data = item.get("tokens", [])
            
            if not image_path or not full_text:
                logger.warning(f"Skipping OCR entry due to missing image_path or full_text: {item}")
                continue

            # 1. Convert raw tokens to Pydantic OCRToken (Strict check)
            validated_tokens: List[OCRToken] = []
            for token_data in tokens_data:
                try:
                    # OCRToken validation ensures text is not empty and BBox is valid [0, 1]
                    validated_tokens.append(OCRToken(**token_data))
                except ValidationError as e:
                    # Hardening: Filter out individual bad tokens
                    logger.warning(f"Invalid OCR Token detected for {image_path}: {e}. Raw data: {token_data}")
            
            # 2. Create the overall OCRLabel
            if validated_tokens:
                try:
                    validated_label = OCRLabel(
                        image_path=image_path,
                        full_text=full_text,
                        tokens=validated_tokens
                    )
                    annotated_labels.append(validated_label)
                except ValidationError as e:
                    # Hardening: Catch failure if aggregated data (e.g., full_text) is invalid
                    logger.error(f"Failed to create valid OCRLabel for {image_path}: {e}")
                
        return annotated_labels