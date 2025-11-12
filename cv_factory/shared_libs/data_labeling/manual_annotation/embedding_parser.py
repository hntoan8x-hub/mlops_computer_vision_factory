# shared_libs/data_labeling/manual_annotation/embedding_parser.py (NEW)

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import EmbeddingLabel
from ....data_labeling.configs.labeler_config_schema import EmbeddingLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class EmbeddingParser(BaseManualAnnotator):
    """
    Specialized Parser for Embedding labels: Handles CSV/DataFrame/Parquet data containing 
    target IDs and the associated feature vector (usually stored as a serialized string or list).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the parser and validates its configuration."""
        super().__init__(config)
        try:
            # Tôn trọng cấu trúc khởi tạo của bạn
            self.parser_config = EmbeddingLabelerConfig(**config) 
        except ValidationError as e:
            logger.critical(f"EmbeddingParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")
            
    def parse(self, raw_input: pd.DataFrame) -> List[StandardLabel]:
        """
        Parses the raw DataFrame and creates validated EmbeddingLabel objects.
        
        Args:
            raw_input: The raw label data as a Pandas DataFrame.

        Returns:
            List[StandardLabel]: List of validated EmbeddingLabel objects.
        """
        if not isinstance(raw_input, pd.DataFrame):
            raise TypeError("EmbeddingParser requires a Pandas DataFrame input.")
            
        annotated_labels: List[EmbeddingLabel] = []
        
        # Giả định các cột cần thiết (cần được định nghĩa trong config schema)
        id_col = "target_id_column" 
        vec_col = "vector_column" 

        # Kiểm tra sự tồn tại của các cột (cần thêm vào schema config)
        if id_col not in raw_input.columns or vec_col not in raw_input.columns:
            logger.warning(f"Required columns ({id_col}, {vec_col}) missing from DataFrame.")
            return []

        for _, row in raw_input.iterrows():
            # NOTE: Giả định vector được lưu dưới dạng chuỗi JSON hoặc list Python trong cột
            raw_vector = row[vec_col] if isinstance(row[vec_col], list) else eval(row[vec_col])
            
            sample_data = {
                "image_path": row["image_path_column"], # Giả định cột này tồn tại
                "target_id": row[id_col],
                "vector": raw_vector
            }
            try:
                validated_label = EmbeddingLabel(**sample_data)
                # Hardening: Kiểm tra kích thước vector
                if len(validated_label.vector) != self.parser_config.vector_dim:
                    raise ValueError("Vector dimension mismatch.")
                
                annotated_labels.append(validated_label)
            except ValidationError as e:
                logger.warning(f"Skipping invalid embedding entry (ID: {row[id_col]}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error during embedding parsing: {e}")
                
        return annotated_labels