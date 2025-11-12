# shared_libs/data_labeling/implementations/pointcloud_labeler.py (NEW)

import logging
from typing import Dict, Any, List, Union
from torch import tensor, float32, long
import pandas as pd
import numpy as np
import os
from pydantic import ValidationError

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import PointCloudLabel
from ...data_labeling.configs.labeler_config_schema import PointCloudLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class PointCloudLabeler(BaseLabeler):
    """
    Concrete Labeler for Point Cloud Processing (3D Detection/Segmentation). 
    
    Orchestrates loading 3D labels (e.g., KITTI, NuScenes format) and converts 
    them into standardized PyTorch Tensors for training.

    Attributes:
        annotation_mode (str): Current mode ("manual" or "auto").
        auto_annotator (Any): Instance for generating 3D proposals.
        config_params (PointCloudLabelerConfig): The validated specific configuration.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        """
        Initializes the PointCloudLabeler and validates its configuration.
        """
        super().__init__(connector_id, config)
        self.annotation_mode: str = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        if not self.validated_config or not isinstance(self.validated_config.params, PointCloudLabelerConfig):
             raise RuntimeError("PointCloudLabeler requires a valid PointCloudLabelerConfig in 'params'.")
             
        self.config_params: PointCloudLabelerConfig = self.validated_config.params 
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., PointCloudProposalAnnotator) if needed."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "pointcloud_processing") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads 3D label data (Manual mode) or metadata (Auto mode).
        
        Returns:
            List[Dict[str, Any]]: List of standardized labels or metadata.
        """
        source_uri = self.config_params.label_source_uri
        
        # 1. Load Raw Data/Metadata (Giả định là DataFrame index hoặc raw JSON/XML)
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # CHẾ ĐỘ MANUAL: Parsing file nhãn 3D
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="pointcloud_processing",
                    config=self.validated_config.model_dump()
                )
                # Parser trả về List[PointCloudLabel] (Pydantic objects)
                validated_labels_pydantic: List[PointCloudLabel] = parser.parse(raw_data)
                
                # Convert Pydantic object to dictionary for DataLoader
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"PointCloud manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("pointclouds", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validates if the sample is ready for training (checks for 3D BBoxes or Point Mask path).
        """
        if not sample.get("pointcloud_path"):
            logger.warning(f"Sample skipped: Missing 'pointcloud_path'.")
            return False
            
        # Hardening: Check if the sample has at least one type of required 3D annotation
        if not sample.get("bbox_3d") and not sample.get("point_mask_path"):
             logger.warning("Sample skipped: Missing both 'bbox_3d' and 'point_mask_path'.")
             return False
            
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> Union[tensor, Dict[str, tensor]]:
        """
        Converts 3D Bounding Boxes and/or Point Segmentation Masks into PyTorch Tensors.
        """
        bbox_3d = label_data.get("bbox_3d")
        point_mask_path = label_data.get("point_mask_path")
        
        tensors: Dict[str, tensor] = {}

        # 1. Convert 3D BBox (x, y, z, l, w, h, yaw)
        if bbox_3d:
            try:
                # BBox 3D thường là List[List[float]] (n_objects, 7)
                tensors['bbox_3d'] = tensor(bbox_3d, dtype=float32)
            except Exception as e:
                logger.error(f"Failed to convert 3D BBox to tensor: {e}")

        # 2. Convert Point Segmentation Mask (chỉ số lớp cho từng điểm)
        if point_mask_path:
            # Logic: Load file nhãn (thường là .label hoặc .npy)
            try:
                # Giả định một tiện ích (ví dụ: np.load) để đọc file nhãn điểm
                # point_mask_array = np.load(point_mask_path).astype(np.int64)
                point_mask_array = np.zeros(100).astype(np.int64) # Placeholder
                tensors['point_mask'] = tensor(point_mask_array).long()
            except Exception as e:
                logger.error(f"Failed to load/convert point mask: {e}")

        if not tensors:
             raise ValueError("PointCloud Labeler failed to convert any valid tensor target.")
             
        # Trả về Dict Tensors (đa mục tiêu)
        return tensors