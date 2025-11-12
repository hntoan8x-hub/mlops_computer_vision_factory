# shared_libs/data_labeling/manual_annotation/keypoint_parser.py (NEW)

import pandas as pd
import logging
from typing import List, Dict, Any, Union
from .base_manual_annotator import BaseManualAnnotator, StandardLabel
from ....data_labeling.configs.label_schema import KeypointLabel, KeypointObject
from ....data_labeling.configs.labeler_config_schema import KeypointLabelerConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class KeypointParser(BaseManualAnnotator):
    """
    Specialized Parser for Keypoint Estimation labels: Handles JSON/XML formats 
    containing normalized keypoint coordinates.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            self.parser_config = KeypointLabelerConfig(**config) 
        except ValidationError as e:
            logger.critical(f"KeypointParser configuration is invalid: {e}")
            raise RuntimeError(f"Invalid Parser Config: {e}")

    def parse(self, raw_input: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[StandardLabel]:
        """
        Parses raw data (DataFrame or List[Dict]) and creates KeypointLabel objects.
        
        Args:
            raw_input: The raw label data.

        Returns:
            List[StandardLabel]: List of validated KeypointLabel objects.
        """
        if isinstance(raw_input, pd.DataFrame):
            raw_list = raw_input.to_dict('records')
        elif isinstance(raw_input, list):
            raw_list = raw_input
        else:
            raise TypeError("KeypointParser requires a Pandas DataFrame or List[Dict] input.")
            
        annotated_labels: List[KeypointLabel] = []
        
        for item in raw_list:
            image_path = item.get("image_path")
            raw_objects = item.get("objects", []) # List of objects containing keypoints
            
            if not image_path or not raw_objects:
                continue

            validated_objects: List[KeypointObject] = []
            for obj_data in raw_objects:
                try:
                    # KeypointObject validation ensures coordinates are normalized
                    validated_objects.append(KeypointObject(**obj_data))
                except ValidationError as e:
                    logger.warning(f"Skipping invalid Keypoint Object for {image_path}: {e}")
            
            if validated_objects:
                try:
                    validated_label = KeypointLabel(
                        image_path=image_path,
                        objects=validated_objects
                    )
                    annotated_labels.append(validated_label)
                except ValidationError as e:
                    logger.error(f"Failed to create valid KeypointLabel for {image_path}: {e}")
                
        return annotated_labels