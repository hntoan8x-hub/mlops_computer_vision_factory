# shared_libs/data_processing/text_components/text_processor_factory.py
import logging
from typing import Dict, Any, Type, Optional
from .base_text_processor import BaseTextProcessor

# Import Atomic Components
from .text_loader import TextLoader
from .text_tokenizer import TextTokenizer
from .text_augmenter import TextAugmenter
from .text_validator import TextValidator

logger = logging.getLogger(__name__)

class TextProcessorFactory:
    """
    Factory class để tạo các instance của Text Processor.
    """
    _PROCESSOR_MAP: Dict[str, Type[BaseTextProcessor]] = {
        "text_loader": TextLoader, 
        "text_tokenizer": TextTokenizer,
        "text_augmenter": TextAugmenter,
        "text_validator": TextValidator,
    }

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> BaseTextProcessor:
        
        config = config or {}
        component_type_lower = component_type.lower()
        key = component_type_lower

        processor_cls = cls._PROCESSOR_MAP.get(key)
        
        if not processor_cls:
            supported = ", ".join(set(cls._PROCESSOR_MAP.keys()))
            raise ValueError(f"Unknown Text component: {component_type}. Supported: {supported}")

        logger.info(f"Creating instance of {processor_cls.__name__} for key '{key}'...")
        try:
            # Giả định Factory nhận config đã được extract từ Pydantic
            return processor_cls(config.get('params', {}))
        except Exception as e:
            raise RuntimeError(f"Factory instantiation failed for '{component_type}': {e}")