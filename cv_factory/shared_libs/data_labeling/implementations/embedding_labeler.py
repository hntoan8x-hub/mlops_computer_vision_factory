# shared_libs/data_labeling/implementations/embedding_labeler.py (Cập nhật)

import logging
from typing import Dict, Any, List, Union, Tuple, Literal
from torch import tensor, long
import pandas as pd

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.label_schema import EmbeddingLabel
from ...data_labeling.configs.labeler_config_schema import EmbeddingLabelerConfig
from ..manual_annotation.factory import ManualAnnotatorFactory
from ..auto_annotation.factory import AutoAnnotatorFactory

logger = logging.getLogger(__name__)

class EmbeddingLabeler(BaseLabeler):
    """
    Concrete Labeler cho Embedding Learning. Hỗ trợ Manual Parsing (tải metadata) 
    và Auto Proposal (sinh vector/ID).
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.id_to_int_map: Dict[str, int] = {"person_A": 1, "person_B": 2} # Ví dụ ánh xạ
        self.annotation_mode: Literal["manual", "auto"] = self.validated_config.get("annotation_mode", "manual")
        self.auto_annotator = self._initialize_auto_annotator()

    def _initialize_auto_annotator(self):
        """Khởi tạo Auto Annotator (ví dụ: EmbeddingProposalAnnotator) nếu cần."""
        if self.annotation_mode == "auto":
             auto_config = self.validated_config.model_dump().get("auto_annotation", {})
             if auto_config:
                 annotator_type = auto_config.get("annotator_type", "embedding") 
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
            # CHẾ ĐỘ MANUAL: Parsing file nhãn (thường là CSV/JSON của target_id)
            try:
                parser = ManualAnnotatorFactory.get_annotator(
                    domain_type="embedding", # Cần đảm bảo EmbeddedParser đã được triển khai
                    config=self.validated_config.model_dump()
                )
                validated_labels_pydantic: List[EmbeddingLabel] = parser.parse(raw_data)
                final_labels = [label.model_dump() for label in validated_labels_pydantic]
            except Exception as e:
                logger.error(f"Embedding manual parsing failed: {e}")
                raise
            
        elif self.annotation_mode == "auto":
            # CHẾ ĐỘ AUTO: Raw data là List[Dict] metadata ảnh
            # Nhãn sẽ được sinh trong __getitem__
            final_labels = raw_data if isinstance(raw_data, list) else raw_data.get("images", [])
            logger.info(f"Loaded {len(final_labels)} samples for Auto Annotation.")
        else:
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")

        self.raw_labels = final_labels
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Kiểm tra nhãn phải có target_id."""
        target_id = sample.get("target_id")
        return target_id is not None

    def convert_to_tensor(self, label_data: Dict[str, Any]) -> Union[tensor, Dict[str, tensor]]:
        """Chuyển đổi Target ID thành PyTorch Long Tensor ID."""
        target_id = label_data["target_id"]
        
        if isinstance(target_id, str):
            id_int = self.id_to_int_map.get(target_id, -1)
        elif isinstance(target_id, int):
            id_int = target_id
        else:
            raise TypeError("Target ID must be string or integer.")
            
        return tensor(id_int).long()