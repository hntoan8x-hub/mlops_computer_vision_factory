# shared_libs/data_processing/mask_components/mask_processor_factory.py
import logging
from typing import Dict, Any, Type, Optional
from .base_mask_processor import BaseMaskProcessor

# Import Atomic Components (Mô phỏng)
# Cần tạo các file này trong quá trình triển khai thực tế
# from .mask_loader import MaskLoader 
# from .mask_normalizer import MaskNormalizer
# from .mask_augmenter import MaskAugmenter
# from .mask_validator import MaskValidator

logger = logging.getLogger(__name__)

# MOCK CLASSES cho mục đích Factory
class MaskLoader(BaseMaskProcessor):
    def process(self, rgb_image, mask_data): 
        logger.debug("MaskLoader processing...")
        return {"rgb": rgb_image, "mask": mask_data}

class MaskNormalizer(BaseMaskProcessor):
    def process(self, rgb_image, mask_data): 
        logger.debug("MaskNormalizer processing (e.g., One-Hot Encoding)...")
        return {"rgb": rgb_image, "mask": mask_data}

class MaskAugmenter(BaseMaskProcessor):
    def process(self, rgb_image, mask_data): 
        logger.debug("MaskAugmenter processing (e.g., Sync Flip)...")
        return {"rgb": rgb_image, "mask": mask_data}

class MaskValidator(BaseMaskProcessor):
    def process(self, rgb_image, mask_data): 
        logger.debug("MaskValidator processing (e.g., Check Class Distribution)...")
        return {"rgb": rgb_image, "mask": mask_data}

class MaskProcessorFactory:
    """
    Factory class để tạo các instance của Mask Processor.
    """
    _PROCESSOR_MAP: Dict[str, Type[BaseMaskProcessor]] = {
        "loader": MaskLoader, 
        "normalizer": MaskNormalizer,
        "augmenter": MaskAugmenter,
        "validator": MaskValidator,
    }

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> BaseMaskProcessor:
        
        config = config or {}
        component_type_lower = component_type.lower()
        key = component_type_lower
        
        # Mapping tên ngắn
        if component_type_lower.endswith('_loader'): key = 'loader'
        elif component_type_lower.endswith('_normalizer'): key = 'normalizer'
        elif component_type_lower.endswith('_augmenter'): key = 'augmenter'
        elif component_type_lower.endswith('_validator'): key = 'validator'

        processor_cls = cls._PROCESSOR_MAP.get(key)
        
        if not processor_cls:
            supported = ", ".join(set(cls._PROCESSOR_MAP.keys()))
            raise ValueError(f"Unknown mask component: {component_type}. Supported: {supported}")

        logger.info(f"Creating instance of {processor_cls.__name__} for key '{key}'...")
        try:
            return processor_cls(config.get('params', {})) # Giả định config được truyền qua Pydantic
        except Exception as e:
            raise RuntimeError(f"Factory instantiation failed for '{component_type}': {e}")