import logging
import numpy as np
import torch
from typing import Dict, Any, Union
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from shared_libs.ml_core.evaluator.base.base_explainer import BaseExplainer

logger = logging.getLogger(__name__)

class IGExplainer(BaseExplainer):
    """
    A concrete explainer for visualizing model decisions using Integrated Gradients.
    
    This explainer attributes the model's output to its input features, showing
    which pixels are most important for a prediction.
    """
    
    def __init__(self, model: torch.nn.Module, **kwargs: Dict[str, Any]):
        """
        Initializes the IGExplainer.
        """
        self.model = model
        self.ig = IntegratedGradients(self.model)
        logger.info("Initialized IGExplainer.")
        
    def explain(self, model: Any, image: Union[np.ndarray, torch.Tensor], 
                target_class: int, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Generates Integrated Gradients attributions for a single image.

        Args:
            model (Any): The PyTorch model.
            image (Union[np.ndarray, torch.Tensor]): The input image tensor (C x H x W).
            target_class (int): The index of the class to explain.
            **kwargs: Additional parameters.

        Returns:
            torch.Tensor: The attribution scores for each pixel.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a PyTorch tensor.")

        # Add a batch dimension if missing
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        try:
            attributions = self.ig.attribute(image, target=target_class, **kwargs)
            return attributions.squeeze(0)
        except Exception as e:
            logger.error(f"Failed to generate Integrated Gradients explanation: {e}")
            raise

    def visualize(self, explanation: torch.Tensor, image: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Visualizes the Integrated Gradients attributions on top of the original image.

        Args:
            explanation (torch.Tensor): The attribution tensor from the `explain` method.
            image (np.ndarray): The original image (H x W x C, 0-255 RGB).
            **kwargs: Additional parameters for visualization.

        Returns:
            np.ndarray: The visualized image with the explanation overlay.
        """
        try:
            # We use a Captum utility function for clean visualization
            img_with_attributions = viz.visualize_image_attr(
                explanation.permute(1, 2, 0).cpu().numpy(),
                image,
                method='blended_heat_map',
                sign='all',
                show_colorbar=False,
                cmap='RdGn',
                **kwargs
            )
            # The function returns a tuple (figure, axes). We want the numpy array of the image.
            return img_with_attributions[0]
        except Exception as e:
            logger.error(f"Failed to visualize Integrated Gradients: {e}")
            raise