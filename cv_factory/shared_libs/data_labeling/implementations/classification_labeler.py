# shared_libs/data_labeling/implementations/classification_labeler.py (Cập nhật)

import logging
from typing import Dict, Any, List, Union
import pandas as pd
from torch import tensor, long

from ..base_labeler import BaseLabeler
from ...data_labeling.configs.labeler_config_schema import ClassificationLabelerConfig
from ...data_labeling.configs.label_schema import ClassificationLabel

# IMPORT CÁC FACTORY MỚI
from ..manual_annotation.factory import ManualAnnotatorFactory
# from ..auto_annotation.factory import AutoAnnotatorFactory # Không dùng trong Class.
# from ..semi_annotation.semi_annotator_factory import SemiAnnotatorFactory # Không dùng trong Class.

logger = logging.getLogger(__name__)

class ClassificationLabeler(BaseLabeler):
    """
    Concrete Labeler cho Image Classification. 
    Điều phối luồng Manual Parsing và ánh xạ nhãn.
    """

    def __init__(self, connector_id: str, config: Dict[str, Any]):
        super().__init__(connector_id, config)
        self.class_name_to_id: Dict[str, int] = {}
        # Ép kiểu config đã được validate
        self.config: ClassificationLabelerConfig = self.validated_config.params 
        self._load_class_map()
        
    def _load_class_map(self):
        """Tải ánh xạ nhãn (string) sang ID (integer) từ config."""
        # Logic này giữ nguyên để tải map từ file (hoặc sử dụng mặc định)
        self.class_name_to_id = {"dog": 0, "cat": 1, "bird": 2} # Ví dụ

    def load_labels(self) -> List[Dict[str, Any]]:
        """
        Tải dữ liệu nhãn thô và sử dụng Manual Annotator để chuẩn hóa.
        Classification mặc định sử dụng luồng Manual Parsing (đọc file).
        """
        source_uri = self.config.label_source_uri
        
        # 1. I/O: Dùng Connector để tải dữ liệu thô (ví dụ: file CSV)
        try:
            with self.get_source_connector() as connector:
                raw_data = connector.read(source_uri=source_uri)
        except Exception as e:
            logger.error(f"Failed to load raw classification data from {source_uri}: {e}")
            raise

        # 2. Chuẩn hóa: SỬ DỤNG MANUAL ANNOTATOR FACTORY
        try:
            # Lấy ClassificationParser thông qua Factory
            parser = ManualAnnotatorFactory.get_annotator(
                domain_type="classification", 
                config=self.validated_config.model_dump()
            )
            # Parser trả về List[ClassificationLabel] (Pydantic objects)
            validated_labels_pydantic: List[ClassificationLabel] = parser.parse(raw_data)
        except Exception as e:
            logger.error(f"Classification manual parsing failed: {e}")
            raise
            
        # 3. Chuyển Pydantic objects về Dict và cập nhật class map
        final_labels = []
        for label_obj in validated_labels_pydantic:
            label_dict = label_obj.model_dump()
            label_name = str(label_dict['label'])
            
            # Cập nhật ánh xạ ID (nếu nhãn mới được tìm thấy trong dữ liệu)
            if label_name not in self.class_name_to_id:
                self.class_name_to_id[label_name] = len(self.class_name_to_id)
                
            final_labels.append(label_dict)

        self.raw_labels = final_labels
        logger.info(f"Classification Labeler loaded {len(self.raw_labels)} samples.")
        return self.raw_labels

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        # Giữ nguyên logic validation
        label = str(sample.get("label"))
        return label in self.class_name_to_id

    def convert_to_tensor(self, label_data: Dict[str, Any]):
        # Giữ nguyên logic chuyển đổi Tensor
        label_name = str(label_data["label"])
        label_id = self.class_name_to_id.get(label_name, -1)
        if label_id == -1:
            raise ValueError(f"Label '{label_name}' not found in class map during conversion.")
        return tensor(label_id).long()