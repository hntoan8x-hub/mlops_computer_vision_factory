# cv_factory/shared_libs/ml_core/pipeline_components_cv/factories/component_factory.py (UPDATED)

import logging
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.pipeline_components_cv.base.base_component import BaseComponent
from shared_libs.ml_core.pipeline_components_cv.configs.component_config_schema import PipelineStepConfig

# --- Import các Adapter CV CORE (Pre-existing) ---
# Cleaning/Preprocessing Adapters
from shared_libs.ml_core.pipeline_components_cv._cleaner.cv_resizer import CVResizer
from shared_libs.ml_core.pipeline_components_cv._cleaner.cv_normalizer import CVNormalizer
from shared_libs.ml_core.pipeline_components_cv._cleaner.cv_color_space_cleaner import CVColorSpaceCleaner 

# Augmentation Adapters
from shared_libs.ml_core.pipeline_components_cv._augmenter.cv_flip_rotate import CVFlipRotate
from shared_libs.ml_core.pipeline_components_cv._augmenter.cv_noise_injection import CVNoiseInjection
from shared_libs.ml_core.pipeline_components_cv._augmenter.cv_cutmix import CVCutMix
from shared_libs.ml_core.pipeline_components_cv._augmenter.cv_mixup import CVMixup

# Feature Extraction/Dimensionality Reduction Adapters
from shared_libs.ml_core.pipeline_components_cv._feature_extractor.cv_dim_reducer import CVDimReducer
from shared_libs.ml_core.pipeline_components_cv._feature_extractor.cv_hog_extractor import CVHOGExtractor
from shared_libs.ml_core.pipeline_components_cv._feature_extractor.cv_sift_extractor import CVSIFTExtractor
from shared_libs.ml_core.pipeline_components_cv._feature_extractor.cv_orb_extractor import CVORBExtractor

# Deep Learning Embedder Adapters
from shared_libs.ml_core.pipeline_components_cv._embedder.cv_cnn_embedder import CVCNNEmbedder
from shared_libs.ml_core.pipeline_components_cv._embedder.cv_vit_embedder import CVViTEmbedder

# --- Import các Adapter DOMAIN-SPECIFIC (MỚI) ---
# Depth Domain Adapters
from shared_libs.ml_core.pipeline_components_cv.depth.cv_depth_cleaner import CVDepthCleaner
from shared_libs.ml_core.pipeline_components_cv.depth.cv_depth_augmenter import CVDepthAugmenter

# Mask Domain Adapters
from shared_libs.ml_core.pipeline_components_cv.mask.cv_mask_cleaner import CVMaskCleaner
from shared_libs.ml_core.pipeline_components_cv.mask.cv_mask_augmenter import CVMaskAugmenter

# Point Cloud Domain Adapters
from shared_libs.ml_core.pipeline_components_cv.pointcloud.cv_pointcloud_cleaner import CVPointCloudCleaner
from shared_libs.ml_core.pipeline_components_cv.pointcloud.cv_pointcloud_augmenter import CVPointCloudAugmenter

# Text Domain Adapters
from shared_libs.ml_core.pipeline_components_cv.text.cv_text_tokenizer import CVTextTokenizer
from shared_libs.ml_core.pipeline_components_cv.text.cv_text_augmenter import CVTextAugmenter

logger = logging.getLogger(__name__)

class ComponentFactory:
    """
    A factory class for creating different types of pipeline components (Adapters).
    """
    _COMPONENT_MAP: Dict[str, Type[BaseComponent]] = {
        # CORE CV (13)
        "resizer": CVResizer,
        "normalizer": CVNormalizer,
        "color_space": CVColorSpaceCleaner,
        "flip_rotate": CVFlipRotate,
        "noise_injection": CVNoiseInjection,
        "cutmix": CVCutMix,
        "mixup": CVMixup,
        "dim_reducer": CVDimReducer,
        "hog_extractor": CVHOGExtractor,
        "sift_extractor": CVSIFTExtractor,
        "orb_extractor": CVORBExtractor,
        "cnn_embedder": CVCNNEmbedder,
        "vit_embedder": CVViTEmbedder,

        # DOMAIN-SPECIFIC (8) <--- MỚI
        "depth_cleaner": CVDepthCleaner,
        "depth_augmenter": CVDepthAugmenter,
        "mask_cleaner": CVMaskCleaner,
        "mask_augmenter": CVMaskAugmenter,
        "pointcloud_cleaner": CVPointCloudCleaner,
        "pointcloud_augmenter": CVPointCloudAugmenter,
        "text_tokenizer": CVTextTokenizer,
        "text_augmenter": CVTextAugmenter,
    } # Tổng cộng 21 Adapters đã được Hardening

    @classmethod
    def create(cls, component_type: str, config: Optional[Dict[str, Any]] = None) -> BaseComponent:
        """
        Creates and returns a component instance (Adapter) based on its type.
        """
        config = config or {}
        component_type_lower = component_type.lower()
        component_cls = cls._COMPONENT_MAP.get(component_type_lower)
        
        if not component_cls:
            supported_components = ", ".join(cls._COMPONENT_MAP.keys())
            logger.error(f"Unsupported component type: '{component_type}'. Supported types are: {supported_components}")
            raise ValueError(f"Unsupported component type: '{component_type}'. Supported types are: {supported_components}")

        logger.info(f"Creating instance of {component_cls.__name__} (Adapter) from Factory...")
        try:
            # HARDENING: Call the Adapter's constructor with validated parameters
            return component_cls(**config)
        except Exception as e:
            # CRITICAL: Catch errors during instantiation (e.g., bad parameter type/range)
            logger.error(f"Factory failed to instantiate {component_cls.__name__} with config {config}. Error: {e}")
            raise RuntimeError(f"Factory instantiation failed for '{component_type}': {e}")