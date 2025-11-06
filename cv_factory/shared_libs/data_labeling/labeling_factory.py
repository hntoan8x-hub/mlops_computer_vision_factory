# shared_libs/data_labeling/labeling_factory.py (CONFIRMED)

import logging
from typing import Dict, Any, Type

# Import Base Abstraction
from .base_labeler import BaseLabeler

# Import Config Schema để Validate Input
from .configs.labeler_config_schema import LabelerConfig

# Import Concrete Implementations (Sử dụng các file đã cập nhật)
from .implementations.classification_labeler import ClassificationLabeler
from .implementations.detection_labeler import DetectionLabeler
from .implementations.segmentation_labeler import SegmentationLabeler
from .implementations.ocr_labeler import OCRLabeler
from .implementations.embedding_labeler import EmbeddingLabeler

logger = logging.getLogger(__name__)

class LabelingFactory:
    """
    Factory class chịu trách nhiệm tạo ra các instance của BaseLabeler.
    Đây là cổng vào chính cho tầng Data Labeling.
    """
    
    # Ánh xạ task_type (string) sang Labeler Class cụ thể
    LABELER_MAPPING: Dict[str, Type[BaseLabeler]] = {
        "classification": ClassificationLabeler,
        "detection": DetectionLabeler,
        "segmentation": SegmentationLabeler,
        "ocr": OCRLabeler,
        "embedding": EmbeddingLabeler,
    }

    @staticmethod
    def get_labeler(connector_id: str, raw_config: Dict[str, Any]) -> BaseLabeler:
        """
        Khởi tạo và trả về một Labeler cụ thể.

        Args:
            connector_id (str): ID duy nhất cho Labeler.
            raw_config (Dict[str, Any]): Cấu hình thô từ file JSON/YAML.

        Returns:
            BaseLabeler: Một instance của Labeler class tương ứng.
        """
        
        # 1. Validate cấu hình sơ bộ và lấy task_type
        try:
            validated_config = LabelerConfig(**raw_config)
            task_type = validated_config.task_type
        except Exception as e:
            raise ValueError(f"Cấu hình Labeler không hợp lệ cho ID '{connector_id}': {e}")
            
        # 2. Lựa chọn và Khởi tạo Labeler Class
        if task_type not in LabelingFactory.LABELER_MAPPING:
            raise ValueError(
                f"Unsupported labeler task type: '{task_type}'. "
                f"Available types are: {list(LabelingFactory.LABELER_MAPPING.keys())}"
            )
            
        LabelerClass = LabelingFactory.LABELER_MAPPING[task_type]
        
        try:
            # Labeler cấp cao sẽ tự gọi các Factory con (Manual/Auto/Semi) bên trong __init__
            labeler = LabelerClass(
                connector_id=connector_id,
                config=raw_config
            )
            logger.info(f"[{connector_id}] Successfully created {task_type} Labeler.")
            return labeler
        except Exception as e:
            logger.error(f"[{connector_id}] Failed to instantiate {task_type} Labeler: {e}")
            raise RuntimeError(f"Labeler creation failed for type '{task_type}': {e}")