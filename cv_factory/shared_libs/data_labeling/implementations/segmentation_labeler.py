# shared_libs/data_labeling/implementations/segmentation_labeler.py (Cập nhật)

import logging
from typing import Dict, Any, List, Union, Tuple, Literal
from torch import tensor, long
import numpy as np

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import SegmentationLabel
from ...data_labeling.configs.labeler_config_schema import SegmentationLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class SegmentationLabeler(BaseLabeler):
    """
    Concrete Labeler cho Semantic Segmentation. Hỗ trợ Manual Parsing (tải mask) 
    và Auto Proposal (sinh mask).
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.annotation_mode: Literal["manual", "auto"] = self.validated_config.get("annotation_mode", "manual")
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.model_dump().get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "segmentation") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """Tải dữ liệu nhãn hoặc metadata ảnh tùy theo chế độ Annotation."""
        source_uri = self.validated_config.params.label_source_uri
        
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # CHẾ ĐỘ MANUAL: Parsing file danh sách mask
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="segmentation",
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[SegmentationLabel] = parser.parse(raw_data)
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Segmentation manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            # CHẾ ĐỘ AUTO: Raw data là List[Dict] metadata ảnh
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Kiểm tra nhãn phải có mask_path và image_path."""
        return "mask_path" in sample and "image_path" in sample

    def convert_to_tensor(self, label_data: Dict[str, Any]):
        """Tải ảnh mask và chuyển thành PyTorch Long Tensor."""
        mask_path = label_data["mask_path"]
        
        # NOTE: Logic tải mask ảnh từ mask_path và chuyển thành Tensor.
        # Giả định:
        # with Image.open(mask_path) as img:
        #     mask_array = np.array(img.convert('L'), dtype=np.int64)
        
        mask_array = np.zeros((100, 100), dtype=np.int64) # Giả lập mask array
        return tensor(mask_array).long()