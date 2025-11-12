# shared_libs/data_processing/pointcloud_components/pointcloud_processor_factory.py
import logging
from typing import Dict, Any, Type, Optional
from .base_pointcloud_processor import BasePointcloudProcessor

# Import Atomic Components
from .pointcloud_loader import PointcloudLoader
from .pointcloud_normalizer import PointcloudNormalizer
from .pointcloud_augmenter import PointcloudAugmenter
from .pointcloud_voxelizer import PointcloudVoxelizer

logger = logging.getLogger(__name__)

class PointcloudProcessorFactory:
    """
    Factory class để tạo các instance của Point Cloud Processor.
    """
    _PROCESSOR_MAP: Dict[str, Type[BasePointcloudProcessor]] = {
        "pc_loader": PointcloudLoader, 
        "pc_normalizer": PointcloudNormalizer,
        "pc_augmenter": PointcloudAugmenter,
        "pc_voxelizer": PointcloudVoxelizer,
    }

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> BasePointcloudProcessor:
        
        config = config or {}
        component_type_lower = component_type.lower()
        key = component_type_lower
        
        # Xử lý trường hợp tên ngắn (pc_loader)
        if component_type_lower.startswith('pc_'): 
             key = component_type_lower

        processor_cls = cls._PROCESSOR_MAP.get(key)
        
        if not processor_cls:
            supported = ", ".join(set(cls._PROCESSOR_MAP.keys()))
            raise ValueError(f"Unknown Point Cloud component: {component_type}. Supported: {supported}")

        logger.info(f"Creating instance of {processor_cls.__name__} for key '{key}'...")
        try:
            # Giả định Factory nhận config đã được extract từ Pydantic
            return processor_cls(config.get('params', {}))
        except Exception as e:
            raise RuntimeError(f"Factory instantiation failed for '{component_type}': {e}")