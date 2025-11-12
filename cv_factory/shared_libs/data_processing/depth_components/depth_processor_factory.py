# shared_libs/data_processing/depth_components/depth_processor_factory.py
import logging
from typing import Dict, Any, Type, Optional
from .base_depth_processor import BaseDepthProcessor

# Import Atomic Components
from .depth_loader import DepthLoader
from .depth_normalizer import DepthNormalizer
from .depth_augmenter import DepthAugmenter
from .depth_validator import DepthValidator
from .disparity_converter import DisparityConverter

logger = logging.getLogger(__name__)

class DepthProcessorFactory:
    """
    Factory class để tạo các instance của Depth Processor.
    """
    _PROCESSOR_MAP: Dict[str, Type[BaseDepthProcessor]] = {
        "depthloader": DepthLoader,
        "depthnormalizer": DepthNormalizer,
        "depthaugmenter": DepthAugmenter,
        "depthvalidator": DepthValidator,
        "disparityconverter": DisparityConverter,
        # Sử dụng tên component không có tiền tố để khớp với config
        "loader": DepthLoader, 
        "normalizer": DepthNormalizer,
        "augmenter": DepthAugmenter,
        "validator": DepthValidator,
        "converter": DisparityConverter,
    }

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDepthProcessor:
        """
        Tạo và trả về một instance của Depth Processor dựa trên type.
        
        Args:
            component_type (str): Loại component cần tạo (ví dụ: "loader").
            config (Optional[Dict[str, Any]]): Tham số cấu hình cho component.

        Returns:
            BaseDepthProcessor: Instance của processor được yêu cầu.

        Raises:
            ValueError: Nếu component_type không được hỗ trợ.
            RuntimeError: Nếu khởi tạo thất bại.
        """
        config = config or {}
        component_type_lower = component_type.lower()
        
        # Xử lý trường hợp người dùng truyền tên đầy đủ (ví dụ: DepthLoader) hoặc tên ngắn (loader)
        if component_type_lower.endswith('loader'):
            key = 'loader'
        elif component_type_lower.endswith('normalizer'):
            key = 'normalizer'
        elif component_type_lower.endswith('augmenter'):
            key = 'augmenter'
        elif component_type_lower.endswith('validator'):
            key = 'validator'
        elif component_type_lower.endswith('converter'):
            key = 'converter'
        else:
            key = component_type_lower # Dùng tên ngắn trực tiếp từ config YAML

        processor_cls = cls._PROCESSOR_MAP.get(key)
        
        if not processor_cls:
            supported = ", ".join(set(cls._PROCESSOR_MAP.keys()))
            logger.error(f"Unsupported depth component: '{component_type}'. Supported types: {supported}")
            raise ValueError(f"Unknown depth component: {component_type}. Supported: {supported}")

        logger.info(f"Creating instance of {processor_cls.__name__} for key '{key}'...")
        try:
            return processor_cls(config)
        except Exception as e:
            logger.error(f"Factory failed to instantiate {processor_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{component_type}': {e}")