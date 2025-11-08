# shared_libs/data_labeling/implementations/ocr_labeler.py (Hardened)

import logging
from typing import Dict, Any, List, Union, Tuple, Literal
from torch import tensor, long, float32
import pandas as pd
from pydantic import ValidationError
import numpy as np # Used for dummy tensor creation

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import OCRLabel
from ...data_labeling.configs.labeler_config_schema import OCRLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class OCRLabeler(BaseLabeler):
    """
    Concrete Labeler for OCR/Text Extraction tasks. 
    
    Supports Manual Parsing (loading JSON/XML/list files) and Auto Proposal 
    (generating text and BBoxes). Final output is tokenized and padded.

    Attributes:
        annotation_mode (Literal): Current mode ("manual" or "auto").
        auto_annotator (Any): Instance for generating OCR proposals.
        config_params (OCRLabelerConfig): The validated specific configuration.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.annotation_mode: Literal["manual", "auto"] = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        if not self.validated_config or not isinstance(self.validated_config.params, OCRLabelerConfig):
             raise RuntimeError("OCRLabeler requires a valid OCRLabelerConfig in 'params'.")
             
        self.config_params: OCRLabelerConfig = self.validated_config.params 
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., OCRProposalAnnotator) if needed."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "ocr") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads label data (Manual mode) or image metadata (Auto mode).
        
        Returns:
            List[Dict[str, Any]]: List of standardized labels or metadata.
            
        Raises:
            Exception: If raw data loading or parsing fails.
        """
        source_uri = self.config_params.label_source_uri
        
        # 1. Load Raw Data/Metadata
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # CHẾ ĐỘ MANUAL: Parsing file nhãn (JSON/XML)
            try:
                # Cần đảm bảo OCRParser đã được triển khai và đăng ký trong ManualAnnotatorFactory
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="ocr",
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[OCRLabel] = parser.parse(raw_data)
                
                # Hardening: Convert Pydantic object to dictionary for DataLoader
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"OCR manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            # CHẾ ĐỘ AUTO: Raw data là List[Dict] metadata ảnh. Nhãn sinh trong __getitem__.
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validates if the sample is ready for training (checks for full_text and tokens).

        Args:
            sample: The label sample (dictionary format).
            
        Returns:
            bool: True if the sample is valid.
        """
        # Hardening: Check essential fields derived from OCRLabel schema
        if not sample.get("full_text") or not isinstance(sample.get("tokens"), list):
            logger.warning(f"Sample skipped: Missing 'full_text' or 'tokens' list.")
            return False
            
        # Optional: Check if the number of BBoxes matches the number of tokens (words)
        # Note: This check relies on the structure of the tokens list.
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> Tuple[tensor, tensor]:
        """
        Converts text tokens into token ID tensors (padded) and BBox tensors.

        Args:
            label_data: The standardized label data (dictionary format).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded token ID tensor and BBox tensor.
            
        Raises:
            ValueError: If tokenization fails or data is missing.
        """
        tokens: List[Dict[str, Any]] = label_data.get("tokens", [])
        if not tokens:
            return tensor([]).long(), tensor([]).float()

        # Hardening: Use config for max length and padding
        max_len = self.config_params.max_sequence_length
        # Placeholder for Tokenizer/Vocab utility
        
        # 1. Simulate Tokenization/Conversion to IDs
        # In a real scenario, this uses self.config_params.tokenizer_config
        token_ids: List[int] = [hash(t['text']) % 1000 for t in tokens][:max_len]
        
        # 2. Pad Sequence
        padding_id = 0 # Assume 0 is the ID for self.config_params.padding_token
        padded_ids = token_ids + [padding_id] * (max_len - len(token_ids))
        
        # 3. Extract BBoxes
        bboxes: List[Tuple[float, float, float, float]] = [t["bbox"] for t in tokens][:max_len]
        # Pad BBoxes with zeros (or a masked value)
        bbox_padding = [[0.0, 0.0, 0.0, 0.0]] * (max_len - len(bboxes))
        padded_bboxes = bboxes + bbox_padding

        text_tensor = tensor(padded_ids).long()
        bbox_tensor = tensor(padded_bboxes, dtype=float32)
        
        return text_tensor, bbox_tensor