# shared_libs/data_processing/image_components/feature_extractors/image_feature_extractor_factory.py

import logging
from typing import Dict, Any, Type, Optional, Union

# Import Abstractions (Contracts)
from shared_libs.data_processing._base.base_feature_extractor import BaseFeatureExtractor
from shared_libs.data_processing._base.base_embedder import BaseEmbedder 

# Import Atomic Components (Adaptees)
# NOTE: Cần cập nhật đường dẫn chính xác dựa trên cấu trúc cuối cùng.
from shared_libs.data_processing.image_components.feature_extractors.atomic.sift_extractor import SIFTExtractor
from shared_libs.data_processing.image_components.feature_extractors.atomic.orb_extractor import ORBExtractor
from shared_libs.data_processing.image_components.feature_extractors.atomic.hog_extractor import HOGExtractor
# Giả định các Embedders Atomic nằm trong một folder khác hoặc được Factory này quản lý:
from shared_libs.data_processing.image_components.embedders.atomic.cnn_embedder import CNNEmbedder 
from shared_libs.data_processing.image_components.embedders.atomic.vit_embedder import VITEmbedder 

logger = logging.getLogger(__name__)

# Union Type for Factory Output
ComponentBaseType = Union[BaseFeatureExtractor, BaseEmbedder]
ComponentType = Type[ComponentBaseType]


class ImageFeatureExtractorFactory:
    """
    A factory class for creating all types of feature components: classical feature 
    extractors and deep learning embedders.

    This centralizes the creation logic, ensuring the Orchestrator has one point 
    of contact for dependency injection.
    """
    
    # HARDENING: Hợp nhất MAPS cho cả Extractors và Embedders.
    _COMPONENT_MAP: Dict[str, ComponentType] = {
        "sift_extractor": SIFTExtractor, # Đã đổi key từ 'sift' sang 'sift_extractor' để nhất quán
        "orb_extractor": ORBExtractor,
        "hog_extractor": HOGExtractor,
        "cnn_embedder": CNNEmbedder,
        "vit_embedder": VITEmbedder,
    }

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> ComponentBaseType:
        """
        Creates and returns a feature component instance (extractor or embedder).

        Args:
            component_type (str): The type of component to create (e.g., "hog_extractor", "cnn_embedder").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters 
                                                to pass to the component's constructor.

        Returns:
            Union[BaseFeatureExtractor, BaseEmbedder]: An instance of the requested component.

        Raises:
            ValueError: If the specified component_type is not supported.
            RuntimeError: If instantiation fails due to bad parameters.
        """
        config = config or {}
        component_type_lower = component_type.lower()
        component_cls = cls._COMPONENT_MAP.get(component_type_lower)
        
        if not component_cls:
            supported = ", ".join(cls._COMPONENT_MAP.keys())
            logger.error(f"Unsupported feature component type: '{component_type}'. Supported types are: {supported}")
            raise ValueError(f"Unsupported feature component type: '{component_type}'. Supported types are: {supported}")

        logger.info(f"Creating instance of {component_cls.__name__}...")
        try:
            # Hardening: Bọc quá trình khởi tạo để bắt lỗi tham số rõ ràng
            return component_cls(**config)
        except Exception as e:
            logger.error(f"Factory failed to instantiate {component_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{component_type}': {e}")
            
    # NOTE: Không cần build_from_config ở đây, Orchestrator sẽ tự quản lý Pydantic Validation.