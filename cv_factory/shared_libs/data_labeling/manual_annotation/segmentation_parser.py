# shared_libs/data_labeling/manual_annotation/segmentation_parser.py (Hardened)

import logging
from typing import List, Dict, Any, Union
import pandas as pd
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import SegmentationLabel
from ....data_labeling.configs.labeler_config_schema import SegmentationLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class SegmentationParser(BaseManualAnnotator):
    """
    Specialized Parser for Segmentation labels: Handles list files (CSV/JSON) 
    containing image paths and corresponding mask file paths.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the parser and validates its configuration.
        """
        super().__init__(config)
        try:
            self.parser_config = SegmentationLabelerConfig(**config) 
        except ValidationError as e:
            logger.critical(f"SegmentationParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")

    def parse(self, raw_input: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[StandardLabel]:
        """
        Parses raw data (DataFrame/List of Dicts) and creates SegmentationLabel objects.
        
        Args:
            raw_input: The raw label data.

        Returns:
            List[StandardLabel]: List of validated SegmentationLabel objects.

        Raises:
            TypeError: If raw_input type is unsupported.
            ValueError: If required columns are missing.
        """
        if isinstance(raw_input, pd.DataFrame):
            df = raw_input
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        else:
            raise TypeError("SegmentationParser expects a Pandas DataFrame or List[Dict] input.")
            
        annotated_labels: List[SegmentationLabel] = []
        
        # Hardening: Assuming img_col and mask_col are defined in the Pydantic config
        # NOTE: SegmentationLabelerConfig needs these attributes defined.
        img_col = self.parser_config.label_source_uri # Placeholder for image path column
        mask_col = self.parser_config.mask_encoding # Placeholder for mask path column
        
        # Since SegmentationLabelerConfig is simple, we must decide how to get columns.
        # Let's assume standard column names for now based on common practice.
        img_col_name = "image_path"
        mask_col_name = "mask_path"
        
        if img_col_name not in df.columns or mask_col_name not in df.columns:
            raise ValueError(f"Required columns ({img_col_name}, {mask_col_name}) missing from DataFrame.")

        for _, row in df.iterrows():
            sample_data = {
                "image_path": row[img_col_name],
                "mask_path": row[mask_col_name]
            }
            # Add class_name if required by the schema (assuming segmentation type)
            if 'class_name' in row:
                 sample_data['class_name'] = row['class_name']
                 
            try:
                # Use Pydantic Schema to check structure
                validated_label = SegmentationLabel(**sample_data)
                annotated_labels.append(validated_label)
            except ValidationError as e:
                logger.warning(f"Skipping invalid segmentation entry (path: {row[img_col_name]}): {e}")
                
        return annotated_labels