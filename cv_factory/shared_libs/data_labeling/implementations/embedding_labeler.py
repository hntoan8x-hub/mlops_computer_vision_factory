# shared_libs/data_labeling/implementations/embedding_labeler.py (Hardened)

import logging
from typing import Dict, Any, List, Union, Tuple, Literal
from torch import tensor, long, float32
import pandas as pd
from pydantic import ValidationError
import numpy as np

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import EmbeddingLabel
from ...data_labeling.configs.labeler_config_schema import EmbeddingLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class EmbeddingLabeler(BaseLabeler):
    """
    Concrete Labeler for Embedding Learning (e.g., Image Retrieval, Face Recognition). 
    
    Supports Manual Parsing (loading metadata/target IDs) and Auto Proposal 
    (generating feature vectors/target IDs).

    Attributes:
        id_to_int_map (Dict[str, int]): Map of target IDs (string) to integer IDs (for classification tasks).
        annotation_mode (Literal): Current mode ("manual" or "auto").
        auto_annotator (Any): Instance for generating embedding proposals.
        config_params (EmbeddingLabelerConfig): The validated specific configuration.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.id_to_int_map: Dict[str, int] = {"__unknown__": 0, "person_A": 1, "person_B": 2} # Default map
        self.annotation_mode: Literal["manual", "auto"] = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        if not self.validated_config or not isinstance(self.validated_config.params, EmbeddingLabelerConfig):
             raise RuntimeError("EmbeddingLabeler requires a valid EmbeddingLabelerConfig in 'params'.")
             
        self.config_params: EmbeddingLabelerConfig = self.validated_config.params 
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., EmbeddingProposalAnnotator) if needed."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "embedding") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads label data (Manual mode) or image metadata (Auto mode).
        
        Returns:
            List[Dict[str, Any]]: List of standardized labels or metadata.
            
        Raises:
            Exception: If raw data loading or parsing fails.
        """
        source_uri = self.config_params.label_source_uri
        
        # 1. Load Raw Data/Metadata
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # CHẾ ĐỘ MANUAL: Parsing file nhãn (target_id, vector, etc.)
            try:
                # Cần đảm bảo EmbeddedParser đã được triển khai và đăng ký trong ManualAnnotatorFactory
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="embedding", 
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[EmbeddingLabel] = parser.parse(raw_data)
                
                # Hardening: Convert Pydantic object to dictionary for DataLoader
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Embedding manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            # CHẾ ĐỘ AUTO: Raw data là List[Dict] metadata ảnh. Nhãn sinh trong __getitem__.
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validates if the sample is ready for training (checks presence of target_id).

        Args:
            sample: The label sample (dictionary format).
            
        Returns:
            bool: True if the sample is valid.
        """
        # Hardening: Ensure target_id is present and is a recognized type
        target_id = sample.get("target_id")
        if not target_id:
            logger.warning(f"Sample skipped: Missing 'target_id'.")
            return False
            
        if isinstance(target_id, str) and target_id not in self.id_to_int_map:
             logger.warning(f"Target ID '{target_id}' is unknown. Skipping sample.")
             return False
             
        # Further checks for 'vector' can be added if required for specific tasks
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> Union[tensor, Dict[str, tensor]]:
        """
        Converts Target ID (for classification) and/or Feature Vector into PyTorch Tensors.

        Args:
            label_data: The standardized label data (dictionary format).
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: The tensorized label data.
            
        Raises:
            TypeError: If Target ID or Vector type is unexpected.
        """
        target_id = label_data["target_id"]
        feature_vector = label_data.get("vector") # Optional

        # 1. Convert Target ID to Long Tensor (for classification/contrastive loss)
        if isinstance(target_id, str):
            id_int = self.id_to_int_map.get(target_id, 0) # Default to 0 (__unknown__)
        elif isinstance(target_id, int):
            id_int = target_id
        else:
            raise TypeError(f"Target ID type must be string or integer, got {type(target_id)}.")
            
        id_tensor = tensor(id_int).long()
        
        # 2. Convert Feature Vector to Float Tensor (for clustering/retrieval tasks)
        if feature_vector is not None:
             try:
                 # Hardening: Ensure vector is converted safely from list/np.ndarray
                 vector_tensor = tensor(feature_vector, dtype=float32)
                 
                 # Return as Dict for multi-target tasks
                 return {"target_id": id_tensor, "vector": vector_tensor}
             except Exception as e:
                 raise TypeError(f"Failed to convert feature 'vector' to tensor: {e}")
        
        return id_tensor # Return only ID if vector is not present/needed