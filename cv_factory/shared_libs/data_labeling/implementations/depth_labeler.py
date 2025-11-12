# shared_libs/data_labeling/implementations/depth_labeler.py (NEW)

import logging
from typing import Dict, Any, List, Union
from torch import tensor, float32, long
import pandas as pd
import numpy as np
import os
from pydantic import ValidationError
from PIL import Image # Thư viện chung để tải ảnh (Depth Map)

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import DepthLabel
from ...data_labeling.configs.labeler_config_schema import DepthLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class DepthLabeler(BaseLabeler):
    """
    Concrete Labeler for Depth Estimation tasks. 
    
    Orchestrates the manual parsing (loading paths to depth maps) and auto proposal 
    (generating depth maps). The main responsibility is converting the raw depth 
    map file into a standardized float Tensor, applying scaling.

    Attributes:
        annotation_mode (str): Current mode ("manual" or "auto").
        auto_annotator (Any): Instance for generating depth proposals.
        config_params (DepthLabelerConfig): The validated specific configuration.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        """
        Initializes the DepthLabeler and validates its configuration.
        """
        super().__init__(connector_id, config)
        self.annotation_mode: str = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        if not self.validated_config or not isinstance(self.validated_config.params, DepthLabelerConfig):
             raise RuntimeError("DepthLabeler requires a valid DepthLabelerConfig in 'params'.")
             
        self.config_params: DepthLabelerConfig = self.validated_config.params 
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., DepthProposalAnnotator) if needed."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "depth_estimation") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads label data (Manual mode) or image metadata (Auto mode) from the configured URI.
        
        Returns:
            List[Dict[str, Any]]: List of standardized labels or metadata.
            
        Raises:
            Exception: If raw data loading or parsing fails.
        """
        source_uri = self.config_params.label_source_uri
        
        # 1. Load Raw Data/Metadata (Giả định là DataFrame hoặc List[Dict] index)
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # CHẾ ĐỘ MANUAL: Parsing file danh sách path
            try:
                # Get Parser (Giả định DepthParser đã được triển khai và đăng ký trong ManualAnnotatorFactory)
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="depth_estimation",
                    config=self.validated_config.model_dump()
                )
                # Parser trả về List[DepthLabel] (Pydantic objects)
                validated_labels_pydantic: List[DepthLabel] = parser.parse(raw_data)
                
                # Convert Pydantic object to dictionary for DataLoader
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Depth manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            # CHẾ ĐỘ AUTO: Tải metadata ảnh. Nhãn được sinh trong __getitem__ (xảy ra trong Orchestrator)
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validates if the sample is ready for training (checks for depth_path).

        Args:
            sample: The label sample (dictionary format).
            
        Returns:
            bool: True if the sample is valid.
        """
        depth_path = sample.get("depth_path")
        # Hardening: Check if depth_path is provided and is a string
        if not depth_path or not isinstance(depth_path, str):
            logger.warning(f"Sample skipped: Missing or invalid 'depth_path'.")
            return False
            
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> tensor:
        """
        Loads the raw depth map file, applies scaling, and converts it into a PyTorch Float Tensor.

        Args:
            label_data: The standardized label data (dictionary format).
            
        Returns:
            torch.Tensor: The depth map as a Float Tensor (H, W).
            
        Raises:
            IOError: If depth map file loading fails.
        """
        depth_path = label_data["depth_path"]
        scale_factor = self.config_params.depth_scale_factor
        
        # 1. Logic tải Depth Map (sử dụng PIL/NumPy/Connector)
        try:
            # Giả định tải 16-bit PNG hoặc Tiff
            with Image.open(depth_path) as img:
                 # Convert to NumPy array (giữ nguyên dtype, thường là uint16)
                 depth_array_raw = np.array(img, dtype=np.uint16)

            # 2. Áp dụng Scaling và chuyển sang Float (đơn vị: meters)
            depth_array_float = depth_array_raw.astype(np.float32) / scale_factor
            
            # 3. Chuyển sang PyTorch Float Tensor
            # NOTE: Tensor Depth Map thường được trả về ở định dạng [H, W] (không có kênh)
            return tensor(depth_array_float).float()
            
        except FileNotFoundError as e:
             raise IOError(f"Failed to load depth map: {e}")
        except Exception as e:
             raise IOError(f"Error converting depth map {depth_path} to tensor: {e}")