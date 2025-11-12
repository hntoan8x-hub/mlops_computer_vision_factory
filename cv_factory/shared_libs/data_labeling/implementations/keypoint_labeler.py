# shared_libs/data_labeling/implementations/keypoint_labeler.py (NEW)

import logging
from typing import Dict, Any, List, Union, Tuple
from torch import tensor, float32, long
import numpy as np
from pydantic import ValidationError

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import KeypointLabel, KeypointObject
from ...data_labeling.configs.labeler_config_schema import KeypointLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class KeypointLabeler(BaseLabeler):
    """
    Concrete Labeler for Keypoint Estimation (Pose Estimation). 
    
    Orchestrates loading Keypoint annotations (e.g., COCO format) and converts 
    them into standardized PyTorch Tensors (coordinates and visibility masks).

    Attributes:
        annotation_mode (str): Current mode ("manual" or "auto").
        auto_annotator (Any): Instance for generating keypoint proposals.
        config_params (KeypointLabelerConfig): The validated specific configuration.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        """
        Initializes the KeypointLabeler and validates its configuration.
        """
        super().__init__(connector_id, config)
        self.annotation_mode: str = self.validated_config.raw_config.get("annotation_mode", "manual")
        
        # Hardening: Ép kiểu config params đã được validate
        if not self.validated_config or not isinstance(self.validated_config.params, KeypointLabelerConfig):
             raise RuntimeError("KeypointLabeler requires a valid KeypointLabelerConfig in 'params'.")
             
        self.config_params: KeypointLabelerConfig = self.validated_config.params 
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Initializes Auto Annotator (e.g., KeypointProposalAnnotator) if needed."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.raw_config.get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "keypoint_estimation") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Loads Keypoint annotations (Manual mode) or metadata (Auto mode).
        
        Returns:
            List[Dict[str, Any]]: List of standardized labels or metadata.
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
            # CHẾ ĐỘ MANUAL: Parsing Keypoint annotations (ví dụ: COCO JSON)
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="keypoint_estimation",
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[KeypointLabel] = parser.parse(raw_data)
                
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Keypoint manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validates if the sample is ready for training (checks for required keypoints count).
        """
        objects: List[Dict[str, Any]] = sample.get("objects", [])
        if not objects:
            return False
            
        # Hardening: Check if all objects have the configured number of keypoints
        required_kps = self.config_params.num_keypoints
        for obj in objects:
            if len(obj.get("keypoints", [])) != required_kps:
                logger.warning(f"Object in sample is missing required number of keypoints ({required_kps}).")
                return False
            
        return True

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> Dict[str, tensor]:
        """
        Converts Keypoint coordinates and optional visibility masks into PyTorch Tensors.
        """
        objects: List[Dict[str, Any]] = label_data.get("objects", [])
        if not objects:
            # Trả về tensors rỗng
            return {'keypoints': tensor([]).float(), 'visibility': tensor([]).long()}

        # 1. Tập hợp tất cả Keypoints và Visibility Masks
        all_kps: List[List[float]] = [] # [[x, y], [x, y], ...]
        all_vis: List[int] = []         # [vis1, vis2, ...]
        
        # Giả định: Keypoints đã được chuẩn hóa về [0, 1]
        for obj in objects:
            # Logic: Tách [x, y] và visibility (nếu có)
            for kp_data in obj.get("keypoints", []):
                if len(kp_data) == 3: # [x, y, vis]
                    all_kps.append(kp_data[:2])
                    all_vis.append(int(kp_data[2]))
                else: # [x, y]
                    all_kps.append(kp_data)
                    all_vis.append(1) # Giả định hiển thị nếu không có vis mask
        
        # 2. Convert to Tensors
        kps_tensor = tensor(all_kps, dtype=float32) # [N_kps, 2]
        vis_tensor = tensor(all_vis, dtype=long)    # [N_kps]

        return {'keypoints': kps_tensor, 'visibility': vis_tensor}