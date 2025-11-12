# shared_libs/ml_core/output_adapter/output_adapter_factory.py (FINAL UPDATE)

import logging
from typing import Dict, Any, Type
from .base_output_adapter import BaseOutputAdapter
from .implementations.classification_adapter import ClassificationAdapter
from .implementations.detection_adapter import DetectionAdapter
from .implementations.segmentation_adapter import SegmentationAdapter
from .implementations.ocr_adapter import OCRAdapter
from .implementations.embedding_adapter import EmbeddingAdapter
from .implementations.depth_adapter import DepthAdapter # <<< IMPORT MỚI >>>
from .implementations.pointcloud_adapter import PointCloudAdapter

logger = logging.getLogger(__name__)

class OutputAdapterFactory:
    """
    Factory class quản lý việc tạo ra các Output Adapter chuyên biệt 
    dựa trên task_type.
    """

    ADAPTER_MAPPING: Dict[str, Type[BaseOutputAdapter]] = {
        "classification": ClassificationAdapter,
        "detection": DetectionAdapter,
        "segmentation": SegmentationAdapter,
        "ocr": OCRAdapter,
        "embedding": EmbeddingAdapter,
        "depth_estimation": DepthAdapter, # <<< ĐĂNG KÝ MỚI >>>
        "pointcloud_processing": PointCloudAdapter
    }

    @staticmethod
    def get_adapter(task_type: str, config: Dict[str, Any]) -> BaseOutputAdapter:
        """
        Tạo và trả về một Output Adapter instance.
        """
        task_type = task_type.lower()
        if task_type not in OutputAdapterFactory.ADAPTER_MAPPING:
            raise ValueError(f"Unsupported task type for Output Adapter: {task_type}")

        AdapterClass = OutputAdapterFactory.ADAPTER_MAPPING[task_type]
        
        try:
            return AdapterClass(config)
        except Exception as e:
            logger.error(f"Failed to instantiate {task_type} Output Adapter: {e}")
            raise RuntimeError(f"Output Adapter creation failed: {e}")