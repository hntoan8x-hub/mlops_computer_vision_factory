# shared_libs/data_labeling/implementations/classification_labeler.py (Hardened)

import logging
from typing import Dict, Any, List, Union
import pandas as pd
from torch import tensor, long
from pydantic import ValidationError

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.labeler_config_schema import ClassificationLabelerConfig
from ...data_labeling.configs.label_schema import ClassificationLabel

# IMPORT FACTORY
from ..manual_annotation.factory import ManualAnnotatorFactory

logger = logging.getLogger(__name__)

class ClassificationLabeler(BaseLabeler):
    """
    Concrete Labeler for Image Classification. 
    
    Orchestrates the manual parsing of classification labels (e.g., from CSV) 
    and handles the mapping of class names to integer IDs required for model training.
    
    Attributes:
        class_name_to_id (Dict[str, int]): Map of class names to integer IDs.
        config_params (ClassificationLabelerConfig): The validated specific configuration for classification.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.class_name_to_id: Dict[str, int] = {}
        
        # Hardening: Ép kiểu config params đã được validate
        # Dùng raw_config để đảm bảo logic Pydantic của BaseLabeler được kích hoạt
        if not self.validated_config or not isinstance(self.validated_config.params, ClassificationLabelerConfig):
             raise RuntimeError("ClassificationLabeler requires a valid ClassificationLabelerConfig in 'params'.")
             
        self.config_params: ClassificationLabelerConfig = self.validated_config.params 
        self._load_class_map()
        
    def _load_class_map(self):
        """
        Loads the class name (string) to ID (integer) mapping from the configured path (optional).
        """
        # Logic này giữ nguyên để tải map từ file (hoặc sử dụng mặc định)
        # In production, this loads self.config_params.class_map_path
        if self.config_params.class_map_path:
             # Load map from JSON/YAML file
             logger.info(f"Loading class map from: {self.config_params.class_map_path}")
             # Placeholder:
             self.class_name_to_id = {"dog": 0, "cat": 1, "bird": 2} 
        elif not self.class_name_to_id:
             self.class_name_to_id = {"__background__": 0} # Default with background

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads raw label data (e.g., CSV) and uses the Manual Annotator to parse and validate it.
        
        Returns:
            List[Dict[str, Any]]: List of standardized labels (dictionary format).

        Raises:
            Exception: If raw data loading or manual parsing fails.
        """
        source_uri = self.config_params.label_source_uri
        
        # 1. I/O: Use Connector to load raw data (e.g., CSV content)
        try:
            with self.get_source_connector() as connector:
                # We assume the Connector returns a DataFrame or raw content that the Parser can handle
                raw_data = connector.read(source_uri=source_uri)
        except Exception as e:
            logger.error(f"Failed to load raw classification data from {source_uri}: {e}")
            raise

        # 2. Standardization: USE MANUAL ANNOTATOR FACTORY
        try:
            # Get ClassificationParser via Factory
            parser = ManualAnnotatorFactory.get_annotator(
                domain_type="classification", 
                config=self.config_params.model_dump() # Pass specific validated config params
            )
            # Parser returns List[ClassificationLabel] (Pydantic objects)
            validated_labels_pydantic: List[ClassificationLabel] = parser.parse(raw_data)
        except Exception as e:
            logger.error(f"Classification manual parsing failed: {e}")
            raise
            
        # 3. Convert Pydantic objects to Dict, update class map, and ensure consistency
        final_labels = []
        for label_obj in validated_labels_pydantic:
            label_dict = label_obj.model_dump()
            label_name = str(label_dict['label'])
            
            # Hardening: Update ID mapping dynamically (important for new classes)
            if label_name not in self.class_name_to_id:
                # Assign new sequential ID, excluding 0 if background is already defined
                new_id = len(self.class_name_to_id)
                self.class_name_to_id[label_name] = new_id
                logger.info(f"Dynamically added new class '{label_name}' with ID {new_id}.")
                
            final_labels.append(label_dict)

        self.raw_labels = final_labels
        logger.info(f"Classification Labeler loaded {len(self.raw_labels)} samples.")
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validates if the sample is ready for training (checks class mapping).

        Args:
            sample: The label sample (dictionary format).
            
        Returns:
            bool: True if the sample is valid.
        """
        label_name = str(sample.get("label"))
        # Hardening: Ensure class exists in the map
        if label_name not in self.class_name_to_id:
            logger.warning(f"Label '{label_name}' from sample is not in the loaded class map. Skipping.")
            return False
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> tensor:
        """
        Converts the standardized class label into a PyTorch Long Tensor ID.

        Args:
            label_data: The standardized label data (dictionary format).
            
        Returns:
            torch.Tensor: The class ID as a 0D Long Tensor.
            
        Raises:
            ValueError: If the label is not found in the class map.
        """
        label_name = str(label_data["label"])
        label_id = self.class_name_to_id.get(label_name, -1)
        
        if label_id == -1:
            # This should ideally not happen if validate_sample was called
            raise ValueError(f"Label '{label_name}' not found in class map during tensor conversion.")
            
        return tensor(label_id).long()