# shared_libs/data_labeling/manual_annotation/base_manual_annotator.py (Hardened)

import abc
from typing import List, Dict, Any, Union
# Import Trusted Label Schemas
from ....data_labeling.configs.label_schema import ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel

# Use the standardized type Union (StandardLabel)
StandardLabel = Union[ClassificationLabel, DetectionLabel, SegmentationLabel, OCRLabel]

class BaseManualAnnotator(abc.ABC):
    """
    Abstract Base Class for Annotators processing manually generated labels (Parsers).
    
    Defines the contract for standardizing raw file data (CSV, JSON, XML) 
    into the canonical Pydantic Schema format (Trusted Labels).

    Attributes:
        config (Dict[str, Any]): The raw configuration passed during initialization.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base parser. Subclasses must validate 'config' against 
        their respective Pydantic schemas immediately.

        Args:
            config: The configuration dictionary.
        """
        self.config = config

    @abc.abstractmethod
    def parse(self, raw_input: Any) -> List[StandardLabel]:
        """
        Executes the parsing of raw label data (from file/database) and standardizes it.

        Args:
            raw_input: The raw label data (e.g., Pandas DataFrame, dictionary from JSON file, XML string).

        Returns:
            List[StandardLabel]: A list of validated, standardized labels (Pydantic objects).
        
        Raises:
            TypeError: If the raw_input type is unsupported.
            Exception: If validation against Pydantic schema fails for an entry.
        """
        raise NotImplementedError