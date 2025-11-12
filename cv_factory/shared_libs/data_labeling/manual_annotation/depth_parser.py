# shared_libs/data_labeling/manual_annotation/depth_parser.py (NEW)

import pandas as pd
import logging
from typing import List, Dict, Any
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import DepthLabel
from ....data_labeling.configs.labeler_config_schema import DepthLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class DepthParser(BaseManualAnnotator):
    """
    Specialized Parser for Depth Estimation labels: Handles CSV/DataFrame index files 
    pointing to raw depth map files (e.g., 16-bit PNG, raw NPY files).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            self.parser_config = DepthLabelerConfig(**config) 
        except ValidationError as e:
            logger.critical(f"DepthParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")

    def parse(self, raw_input: pd.DataFrame) -> List[StandardLabel]:
        """
        Parses the raw DataFrame and creates validated DepthLabel objects.
        
        Args:
            raw_input: The raw label data as a Pandas DataFrame.

        Returns:
            List[StandardLabel]: List of validated DepthLabel objects.
        """
        if not isinstance(raw_input, pd.DataFrame):
            raise TypeError("DepthParser requires a Pandas DataFrame input.")
            
        annotated_labels: List[DepthLabel] = []
        
        # Lấy tên cột từ cấu hình đã được validation
        img_col = self.parser_config.image_path_column
        depth_col = self.parser_config.depth_path_column
        
        if img_col not in raw_input.columns or depth_col not in raw_input.columns:
            raise ValueError(f"Required columns ({img_col}, {depth_col}) missing from DataFrame.")

        for _, row in raw_input.iterrows():
            sample_data = {
                "image_path": row[img_col],
                "depth_path": row[depth_col],
                # Giả định các trường metadata khác có sẵn
                "unit": "meter" # Mặc định là meter
            }
            try:
                validated_label = DepthLabel(**sample_data)
                annotated_labels.append(validated_label)
            except ValidationError as e:
                logger.warning(f"Skipping invalid depth entry (path: {row[img_col]}): {e}")
                
        return annotated_labels