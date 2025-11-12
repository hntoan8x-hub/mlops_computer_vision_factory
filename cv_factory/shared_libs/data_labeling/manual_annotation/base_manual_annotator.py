# shared_libs/data_labeling/manual_annotation/base_manual_annotator.py (UPDATED)

import abc
from typing import List, Dict, Any, Union
# Import Trusted Label Schemas
from ....data_labeling.configs.label_schema import (
    ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel, EmbeddingLabel,
    DepthLabel, PointCloudLabel, KeypointLabel # <<< IMPORTS MỚI >>>
)

# Cập nhật Union để bao gồm tất cả các loại nhãn được hỗ trợ
StandardLabel = Union[
    ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel, EmbeddingLabel,
    DepthLabel, PointCloudLabel, KeypointLabel
]

class BaseManualAnnotator(abc.ABC):
    """
    Abstract Base Class for Annotators processing manually generated labels (Parsers).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def parse(self, raw_input: Any) -> List[StandardLabel]:
        """
        Executes the parsing of raw label data and standardizes it to Pydantic objects.

        Args:
            raw_input (Any): The raw label data (e.g., Pandas DataFrame, dictionary from JSON file).

        Returns:
            List[StandardLabel]: A list of validated, standardized Pydantic label objects.
        """
        raise NotImplementedError