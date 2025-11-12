# shared_libs/ml_core/cv_model/implementations/classification_models.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torchvision.models import resnet18, ResNet18_Weights

from shared_libs.ml_core.cv_model.base.base_cv_model import BaseCVModel

class ResNetClassifier(BaseCVModel):
    """
    ResNet-18 model for image classification, inheriting from BaseCVModel.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the ResNet-18 model.
        
        Config expected to contain:
        - num_classes: int
        - pretrained: bool
        """
        super().__init__(config, **kwargs)
        
        num_classes = config.get('num_classes', 1000)
        pretrained = config.get('pretrained', False)
        
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet18(weights=weights)
        
        # Thay thế layer cuối cùng (fully connected) cho số lượng classes mới
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        self.output_layer = nn.Identity() # Placeholder for potential output layer
        
        logger.info(f"Initialized ResNetClassifier (Classes: {num_classes}, Pretrained: {pretrained})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.model(x)
        return self.output_layer(x)

# Có thể thêm các mô hình Classification khác như ViT, EfficientNet tại đây.

class VisionTransformerClassifier(BaseCVModel):
    """
    Dummy Vision Transformer Classifier for illustration.
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        num_classes = config.get('num_classes', 10)
        # Placeholder: Tạo một mô hình tuyến tính đơn giản thay thế ViT phức tạp
        self.linear = nn.Linear(512, num_classes) 
        logger.info(f"Initialized VisionTransformerClassifier (Dummy Model).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Giả định input x đã được flatten hoặc là embeddings
        # (Trong thực tế, cần tiền xử lý phức tạp hơn cho ViT)
        return self.linear(torch.mean(x, dim=[2, 3])) # Giả sử input là [B, C, H, W]