# cv_factory/shared_libs/ml_core/pipeline_components_cv/factories/component_factory.py

import logging
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent

# --- Import all FINALIZED Adapter Components ---

# Cleaning/Preprocessing Adapters (Stateless)
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_resizer import CVResizer
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_normalizer import CVNormalizer
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_color_space_cleaner import CVColorSpaceCleaner 

# Augmentation Adapters (Stateless)
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_flip_rotate import CVFlipRotate
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_noise_injection import CVNoiseInjection
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_cutmix import CVCutMix
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_mixup import CVMixup

# Feature Extraction/Dimensionality Reduction Adapters (Stateful/Stateless)
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_dim_reducer import CVDimReducer # Stateful (PCA/UMAP)
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_hog_extractor import CVHOGExtractor   # Stateless
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_sift_extractor import CVSIFTExtractor # Stateless
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_orb_extractor import CVORBExtractor     # Stateless

# Deep Learning Embedder Adapters (Stateful - Model Management)
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_cnn_embedder import CVCNNEmbedder
from shared_libs.ml_core.pipeline_components_cv.atomic.cv_vit_embedder import CVViTEmbedder

logger = logging.getLogger(__name__)

class ComponentFactory:
    """
    A factory class for creating different types of pipeline components.

    This class centralizes the creation logic, allowing for a config-driven
    approach to building CV processing pipelines using standardized Adapter classes.
    """
    _COMPONENT_MAP: Dict[str, Type[BaseComponent]] = {
        # CLEANING & PREPROCESSING
        "resizer": CVResizer,
        "normalizer": CVNormalizer,
        "color_space": CVColorSpaceCleaner,

        # AUGMENTATION
        "flip_rotate": CVFlipRotate,
        "noise_injection": CVNoiseInjection,
        "cutmix": CVCutMix,
        "mixup": CVMixup,

        # FEATURE EXTRACTION & DIM REDUCTION
        "dim_reducer": CVDimReducer, # (Stateful - PCA/UMAP)
        "hog_extractor": CVHOGExtractor,
        "sift_extractor": CVSIFTExtractor,
        "orb_extractor": CVORBExtractor,

        # DEEP LEARNING EMBEDDERS
        "cnn_embedder": CVCNNEmbedder,
        "vit_embedder": CVViTEmbedder,
        
        # NOTE: The deprecated "augmenter" and "embedder" entries have been removed/replaced
        # by the specific, clearer names above (e.g., "cnn_embedder" instead of generic "embedder").
    }

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> BaseComponent:
        """
        Creates and returns a component instance based on its type.

        Args:
            component_type (str): The type of component to create (e.g., "resizer", "cnn_embedder").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseComponent: An instance of the requested component Adapter.

        Raises:
            ValueError: If the specified component_type is not supported.
        """
        config = config or {}
        component_cls = cls._COMPONENT_MAP.get(component_type.lower())
        
        if not component_cls:
            supported_components = ", ".join(cls._COMPONENT_MAP.keys())
            logger.error(f"Unsupported component type: '{component_type}'. Supported types are: {supported_components}")
            raise ValueError(f"Unsupported component type: '{component_type}'. Supported types are: {supported_components}")

        logger.info(f"Creating instance of {component_cls.__name__} (Adapter) from Factory...")
        return component_cls(**config)