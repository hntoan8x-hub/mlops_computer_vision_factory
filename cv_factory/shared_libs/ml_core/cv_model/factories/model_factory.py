# shared_libs/ml_core/cv_model/factories/model_factory.py

import torch.nn as nn
import logging
from typing import Dict, Any, Type, Optional
from shared_libs.ml_core.cv_model.configs.model_config_schema import ModelConfig # Sử dụng Schema đã validate
from shared_libs.ml_core.cv_model.base.base_cv_model import BaseCVModel # Sử dụng Base Contract

logger = logging.getLogger(__name__)

# --- Giả định Imports cho các Model Cụ Thể (Concrete Implementations) ---
from ..implementations.classification_models import ResNetClassifier, VisionTransformerClassifier
from ..implementations.segmentation_models import UNetSegmentation
from ..implementations.depth_estimation_models import MonocularDepthModel

class ModelFactory:
    """
    Factory for creating PyTorch model instances (torch.nn.Module or BaseCVModel).
    """

    _MODEL_MAP: Dict[str, Type[BaseCVModel]] = {
        # Classification
        "resnet": ResNetClassifier,
        "vit": VisionTransformerClassifier, 
        
        # Segmentation
        "unet": UNetSegmentation,
        
        # Depth Estimation
        "monocular_depth": MonocularDepthModel,
    }

    @classmethod
    def create(cls, config: ModelConfig) -> nn.Module:
        """
        Tạo và trả về một Model instance dựa trên ModelConfig đã validate.
        
        Args:
            config (ModelConfig): Validated Pydantic schema for the model.
            
        Returns:
            nn.Module: The instantiated PyTorch model.
        """
        model_name = config.name.lower()
        model_cls = cls._MODEL_MAP.get(model_name)
        
        if not model_cls:
            supported_models = ", ".join(cls._MODEL_MAP.keys())
            raise ValueError(f"Unsupported model name: '{model_name}'. Supported models: {supported_models}")

        logger.info(f"Creating model instance: {model_cls.__name__} (Pretrained: {config.pretrained})")
        
        try:
            # Pass the full validated ModelConfig object's dictionary to the model's constructor
            # The concrete model implementation will extract its parameters from this config.
            return model_cls(config=config.model_dump())
        except Exception as e:
            logger.error(f"Failed to instantiate model '{model_name}': {e}")
            raise RuntimeError(f"Model creation failed: {e}")