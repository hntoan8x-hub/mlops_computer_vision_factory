# shared_libs/data_labeling/base_labeler.py (UPDATED)

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Union, Optional
from pydantic import ValidationError
from torch import Tensor

from ..data_labeling.configs.labeler_config_schema import LabelerConfig 
from ..data_ingestion.factories.connector_factory import ConnectorFactory 

logger = logging.getLogger(__name__)

class BaseLabeler(ABC):
    """
    Abstract Base Class (ABC) for all Labelers in CV_Factory.
    
    Defines the contract for loading, validating, and standardizing labels. 
    It handles configuration validation and acts as a facade for accessing data connectors.
    """
    
    def __init__(self, connector_id: str, config: Dict[str, Any]):
        """
        Initializes BaseLabeler.

        Args:
            connector_id (str): Unique ID for the Labeler (usually the task name).
            config (Dict[str, Any]): Raw configuration, which will be validated via Pydantic.
        """
        self.labeler_id = connector_id
        self.raw_config = config
        self.validated_config: Optional[LabelerConfig] = None
        self.connector_factory = ConnectorFactory
        
        self._validate_and_parse_config()
        
        self.raw_labels: List[Dict[str, Any]] = []

    def _validate_and_parse_config(self) -> None:
        """
        Uses the Pydantic Schema (LabelerConfig) to check the validity of the configuration.
        
        Raises:
            ValueError: If the configuration fails Pydantic validation.
        """
        try:
            self.validated_config = LabelerConfig(**self.raw_config)
            logger.info(f"[{self.labeler_id}] Configuration validated successfully.")
        except ValidationError as e:
            logger.critical(f"Labeler configuration failed Pydantic validation for {self.labeler_id}:\n{e}")
            raise ValueError(f"Labeler configuration failed Pydantic validation for {self.labeler_id}:\n{e}")

    # --- Interface Methods ---

    @abstractmethod
    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads raw label data from the source (CSV, JSON, XML,...) and returns List[Dict].
        
        Subclasses must implement logic to call Connector.read() and Parser.parse() here.
        
        Returns:
            List[Dict[str, Any]]: List of loaded, parsed, and standardized label samples.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Performs post-Pydantic validation checks on a single loaded label sample 
        (e.g., checking if the image path exists, BBox semantic validity).
        
        Args:
            sample (Dict[str, Any]): A single label sample (Dict format).
            
        Returns:
            bool: True if the sample is valid for use in training.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_tensor(self, label_data: Any) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Standardizes the processed label data into PyTorch Tensor(s) ready for DataLoader.
        
        Args:
            label_data (Any): The standardized label data.
            
        Returns:
            Union[Tensor, Dict[str, Tensor]]: Tensor or Dictionary of Tensors.
        """
        raise NotImplementedError
        
    # --- Helper Method (Abstraction Hardening) ---

    def get_source_connector(self):
        """
        Creates and returns the appropriate Data Connector to read the label file (e.g., CSV/JSON).
        
        Returns:
            BaseDataConnector: The initialized connector instance.
            
        Raises:
            RuntimeError: If config is not validated.
        """
        if not self.validated_config:
             raise RuntimeError("Configuration not initialized/validated.")
        
        # Lấy URI từ config cụ thể
        source_uri = self.validated_config.params.label_source_uri 
        
        # Hardening: Use simple heuristic to select the connector type
        if source_uri.startswith("s3://") or source_uri.startswith("gs://"):
            connector_type = "image" # Default to image connector for cloud file reads
        elif source_uri.startswith("http://") or source_uri.startswith("https://"):
             connector_type = "api" # Use API Connector
        else:
            connector_type = "image" # Default to ImageConnector for local file/path reads
            
        return self.connector_factory.get_connector(
            connector_type=connector_type,
            connector_config=self.validated_config.params.model_dump(), 
            connector_id=f"{self.labeler_id}_label_reader"
        )