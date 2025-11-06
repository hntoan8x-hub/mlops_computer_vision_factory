# shared_libs/data_labeling/manual_annotation/classification_parser.py

import pandas as pd
from typing import List, Dict, Any
import logging
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import ClassificationLabel
from ....data_labeling.configs.labeler_config_schema import ClassificationLabelerConfig

logger = logging.getLogger(__name__)

class ClassificationParser(BaseManualAnnotator):
    """
    Parser chuyên biệt cho Classification: Xử lý file CSV/DataFrame/Parquet.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Giả định config đã được validate (là ClassificationLabelerConfig.params)
        self.parser_config = ClassificationLabelerConfig(**config) 

    def parse(self, raw_input: pd.DataFrame) -> List[StandardLabel]:
        """
        Phân tích DataFrame thô và tạo ClassificationLabel objects.
        """
        if not isinstance(raw_input, pd.DataFrame):
            raise TypeError("ClassificationParser requires a Pandas DataFrame input.")
            
        annotated_labels: List[ClassificationLabel] = []
        
        # Lấy tên cột từ config
        img_col = self.parser_config.image_path_column
        lbl_col = self.parser_config.label_column
        
        for _, row in raw_input.iterrows():
            sample_data = {
                "image_path": row[img_col],
                "label": row[lbl_col]
            }
            try:
                # Sử dụng Pydantic Schema để kiểm tra cấu trúc
                validated_label = ClassificationLabel(**sample_data)
                annotated_labels.append(validated_label)
            except Exception as e:
                logger.warning(f"Skipping invalid classification entry: {e}")
                
        # Trả về List[StandardLabel]
        return annotated_labels