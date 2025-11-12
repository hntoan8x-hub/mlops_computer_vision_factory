# shared_libs/ml_core/cv_model/implementations/segmentation_models.py

import torch
import torch.nn as nn
from typing import Dict, Any

from shared_libs.ml_core.cv_model.base.base_cv_model import BaseCVModel

# NOTE: UNet implementation is complex. We use a simplified Dummy structure for the Factory.

class UNetSegmentation(BaseCVModel):
    """
    Simplified UNet-like model for semantic segmentation.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the UNet model.
        
        Config expected to contain:
        - num_classes: int (số kênh đầu ra)
        - in_channels: int (số kênh đầu vào)
        """
        super().__init__(config, **kwargs)
        
        num_classes = config.get('num_classes', 1)
        in_channels = config.get('in_channels', 3)

        # Placeholder: Encoder
        self.encoder = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Placeholder: Decoder (để đảm bảo đầu ra có 4 chiều)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, num_classes, kernel_size=1) 
        )
        logger.info(f"Initialized UNetSegmentation (Classes: {num_classes}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Đầu ra là tensor [Batch, Num_Classes, H, W]
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x