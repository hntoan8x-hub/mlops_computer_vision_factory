# shared_libs/data_labeling/implementations/detection_labeler.py (Cập nhật)

import json
from torch import tensor, float32, long
from typing import Dict, Any, List, Union, Tuple, Literal
import logging

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import DetectionLabel
from ...data_labeling.configs.labeler_config_schema import DetectionLabelerConfig

# IMPORT CÁC FACTORY MỚI
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory
from ..semi_annotation.semi_annotator_factory import SemiAnnotatorFactory

logger = logging.getLogger(__name__)

class DetectionLabeler(BaseLabeler):
    """
    Concrete Labeler cho Object Detection. 
    Điều phối 3 chế độ: Manual Parsing, Auto Annotation, và Semi-Annotation Refinement.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.class_name_to_id: Dict[str, int] = {"person": 1, "car": 2} # Ví dụ map
        # Lấy chế độ annotation từ config
        self.annotation_mode: Literal["manual", "auto", "semi"] = self.validated_config.get("annotation_mode", "manual")
        # Khởi tạo các Annotator cần thiết (cho chế độ auto/semi)
        self.auto_annotator = self._initialize_auto_annotator()
        self.semi_annotator = self._initialize_semi_annotator()


    def _initialize_auto_annotator(self):
        """Khởi tạo Auto Annotator (ví dụ: ProposalGenerator) nếu cần."""
        if self.annotation_mode in ["auto", "semi"]:
             # Giả định config auto_model là cần thiết cho AutoAnnotator
             auto_config = self.validated_config.model_dump().get("auto_annotation", {})
             if auto_config:
                 # Chúng ta cần biết loại Auto Annotator nào cần dùng (ví dụ: 'proposal_detection')
                 annotator_type = auto_config.get("annotator_type", "detection_proposal") 
                 return AutoAnnotatorFactory.get_annotator(annotator_type, auto_config)
        return None

    def _initialize_semi_annotator(self):
        """Khởi tạo Semi Annotator (ví dụ: RefinementAnnotator) nếu cần."""
        if self.annotation_mode == "semi":
            semi_config = self.validated_config.model_dump().get("semi_annotation", {})
            if semi_config:
                # Chúng ta cần biết phương pháp Semi nào cần dùng (ví dụ: 'refinement')
                method_type = semi_config.get("method_type", "refinement")
                return SemiAnnotatorFactory.get_annotator(method_type, semi_config)
        return None

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Tải dữ liệu nhãn thô hoặc metadata ảnh tùy theo chế độ Annotation.
        """
        config: DetectionLabelerConfig = self.validated_config.params
        source_uri = config.label_source_uri
        
        # 1. Tải Dữ liệu/Metadata thô
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri) 
        except Exception as e:
            logger.error(f"Failed to load raw data/metadata from {source_uri}: {e}")
            raise
        
        if self.annotation_mode == "manual":
            # 2a. CHẾ ĐỘ MANUAL: Sử dụng Manual Annotator để Parsing (COCO/VOC)
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="detection", 
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[DetectionLabel] = parser.parse(raw_data)
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Detection manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode in ["auto", "semi"]:
            # 2b. CHẾ ĐỘ AUTO/SEMI: Dữ liệu thô là danh sách metadata (image_path)
            # Không có nhãn, chỉ có metadata ảnh. Nhãn sẽ được sinh trong __getitem__.
            
            # Giả định raw_data đã là List[Dict] (metadata ảnh)
            if isinstance(raw_data, dict) and "images" in raw_data:
                 # Ví dụ: Dữ liệu thô là file COCO-like chỉ chứa phần 'images'
                 final_labels = raw_data["images"]
            elif isinstance(raw_data, list):
                 final_labels = raw_data
            else:
                 raise TypeError("Auto/Semi mode expects List[Dict] or Dict with 'images' key.")
            
            logger.info(f"Loaded {len(final_labels)} samples for {self.annotation_mode} annotation. Actual annotation will run per-sample.")
            
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")


        self.raw_labels = final_labels
        return self.raw_labels

    # Các phương thức khác (validate_sample, convert_to_tensor) giữ nguyên
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        # Giữ nguyên logic validation (ví dụ: kiểm tra BBox hợp lệ)
        objects = sample.get("objects", [])
        if not objects:
            return False
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            if x1 >= x2 or y1 >= y2:
                return False
        return True

    def convert_to_tensor(self, sample: Dict[str, Any]) -> Tuple[tensor, tensor]:
        # Giữ nguyên logic chuyển đổi Tensor
        objects: List[Dict[str, Any]] = sample.get("objects", [])
        if not objects:
            return tensor([]).float(), tensor([]).long()

        bboxes: List[Tuple[float, float, float, float]] = [obj["bbox"] for obj in objects]
        class_ids: List[int] = [self.class_name_to_id.get(obj["class_name"], 0) for obj in objects] 

        bbox_tensor = tensor(bboxes, dtype=float32)
        class_id_tensor = tensor(class_ids, dtype=long)
        
        return bbox_tensor, class_id_tensor