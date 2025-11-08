# shared_libs/data_labeling/manual_annotation/classification_parser.py (Hardened)

import pandas as pd
from typing import List, Dict, Any, Union
import logging
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import ClassificationLabel
from ....data_labeling.configs.labeler_config_schema import ClassificationLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class ClassificationParser(BaseManualAnnotator):
    """
    Specialized Parser for Classification labels: Handles CSV/DataFrame/Parquet data formats.
    
    Normalizes raw columns into the Pydantic ClassificationLabel schema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the parser and validates its configuration.

        Args:
            config: Configuration dictionary matching ClassificationLabelerConfig.
        """
        super().__init__(config)
        # Hardening: Use Pydantic to strictly validate and parse the config parameters
        try:
            self.parser_config = ClassificationLabelerConfig(**config)
        except ValidationError as e:
            logger.critical(f"ClassificationParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")

    def parse(self, raw_input: pd.DataFrame) -> List[StandardLabel]:
        """
        Parses the raw DataFrame and creates validated ClassificationLabel objects.
        
        Args:
            raw_input: The raw label data as a Pandas DataFrame.

        Returns:
            List[StandardLabel]: List of validated ClassificationLabel objects.

        Raises:
            TypeError: If raw_input is not a DataFrame.
        """
        if not isinstance(raw_input, pd.DataFrame):
            raise TypeError("ClassificationParser requires a Pandas DataFrame input.")
            
        annotated_labels: List[ClassificationLabel] = []
        
        # Get column names from the validated config
        img_col = self.parser_config.image_path_column
        lbl_col = self.parser_config.label_column
        
        # Hardening: Check if required columns exist in the DataFrame
        if img_col not in raw_input.columns or lbl_col not in raw_input.columns:
            raise ValueError(f"Required columns ({img_col}, {lbl_col}) missing from DataFrame.")

        for _, row in raw_input.iterrows():
            sample_data = {
                "image_path": row[img_col],
                "label": row[lbl_col]
            }
            try:
                # Use Pydantic Schema to enforce structure and rules (Trusted Label)
                validated_label = ClassificationLabel(**sample_data)
                annotated_labels.append(validated_label)
            except ValidationError as e:
                # Hardening: Log detailed warning for skipped invalid entries
                logger.warning(f"Skipping invalid classification entry (path: {row[img_col]}, label: {row[lbl_col]}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error during classification entry parsing: {e}")
                
        return annotated_labels