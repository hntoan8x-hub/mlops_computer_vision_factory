# shared_libs/data_labeling/implementations/segmentation_labeler.py (Hardened)

import logging
from typing import Dict, Any, List, Union, Tuple, Literal
from torch import tensor, long
import numpy as np
from PIL import Image # Import PIL for mask loading simulation
import os
from pydantic import ValidationError

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import SegmentationLabel
from ...data_labeling.configs.labeler_config_schema import SegmentationLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class SegmentationLabeler(BaseLabeler):
    """
    Concrete Labeler for Semantic Segmentation tasks. 
    
    Supports Manual Parsing (loading mask path lists) and Auto Proposal (generating masks).
    The main responsibility is to load the mask artifact and convert it to a Long Tensor mask.

    Attributes:
        annotation_mode (Literal): Current mode ("manual" or "auto").
        auto_annotator (Any): Instance for generating segmentation proposals.
        config_params (SegmentationLabelerConfig): The validated specific configuration.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.annotation_mode: Literal["manual", "auto"] = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        if not self.validated_config or not isinstance(self.validated_config.params, SegmentationLabelerConfig):
             raise RuntimeError("SegmentationLabeler requires a valid SegmentationLabelerConfig in 'params'.")
             
        self.config_params: SegmentationLabelerConfig = self.validated_config.params 
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., SegmentationProposalAnnotator) if needed."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "segmentation") 
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
            # CHẾ ĐỘ MANUAL: Parsing file danh sách mask
            try:
                # Cần đảm bảo SegmentationParser đã được triển khai và đăng ký
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="segmentation",
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[SegmentationLabel] = parser.parse(raw_data)
                
                # Hardening: Convert Pydantic object to dictionary for DataLoader
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Segmentation manual parsing failed: {e}")
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
        Validates if the sample is ready for training (checks for mask_path).

        Args:
            sample: The label sample (dictionary format).
            
        Returns:
            bool: True if the sample is valid.
        """
        mask_path = sample.get("mask_path")
        # Hardening: Check if mask_path is provided and is a string
        if not mask_path or not isinstance(mask_path, str):
            logger.warning(f"Sample skipped: Missing or invalid 'mask_path'.")
            return False
            
        # Optional: Add check for mask encoding consistency (e.g., if mask_encoding is 'png_mask', check file extension)
        
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> tensor:
        """
        Loads the mask artifact (e.g., PNG file) and converts it into a PyTorch Long Tensor.

        Args:
            label_data: The standardized label data (dictionary format).
            
        Returns:
            torch.Tensor: The mask as a 2D Long Tensor (H, W).
            
        Raises:
            IOError: If mask file loading fails.
        """
        mask_path = label_data["mask_path"]
        
        # 1. Logic tải mask ảnh (có thể từ S3/GCS qua Connector)
        try:
            # Placeholder for loading the mask file content
            # In a real scenario, use a Data Connector here if mask_path is remote.
            
            # Simulated loading for a local PNG mask:
            # We assume a utility or the ImageConnector can fetch the mask content.
            if not os.path.exists(mask_path):
                 raise FileNotFoundError(f"Mask file not found at: {mask_path}")
                 
            with Image.open(mask_path) as img:
                 # Hardening: Convert to grayscale ('L') and ensure integer type
                 mask_array = np.array(img.convert('L'), dtype=np.int64)
                 
                 # Semantic check: Ensure mask contains valid class IDs (0, 1, 2, ...)
                 if mask_array.max() > 255: # Masks should be <= 255 if using 8-bit PNG
                      logger.warning(f"Mask {mask_path} has unexpected large pixel values.")

            # 2. Convert to PyTorch Long Tensor
            return tensor(mask_array).long()
            
        except FileNotFoundError as e:
             raise IOError(f"Failed to load mask: {e}")
        except Exception as e:
             raise IOError(f"Error converting mask {mask_path} to tensor: {e}")