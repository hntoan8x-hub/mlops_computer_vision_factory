import logging
import numpy as np
import cv2
import torch
from typing import Dict, Any, Union

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from shared_libs.ml_core.evaluator.base.base_explainer import BaseExplainer

logger = logging.getLogger(__name__)

class GradCAMExplainer(BaseExplainer):
    """
    A concrete explainer for visualizing model decisions using Grad-CAM.
    
    This explainer generates a heatmap overlay on an image, highlighting the
    regions that were most influential for the model's prediction.
    """
    
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module], method: str = "GradCAM"):
        """
        Initializes the GradCAMExplainer.

        Args:
            model (torch.nn.Module): The PyTorch model to be explained.
            target_layers (List[torch.nn.Module]): A list of target layers to compute the Grad-CAM on.
            method (str): The specific CAM method to use (e.g., "GradCAM", "GradCAMPlusPlus").
        """
        self.model = model
        self.target_layers = target_layers
        
        cam_methods = {
            "GradCAM": GradCAM,
            "GradCAM++": GradCAMPlusPlus,
            "XGradCAM": XGradCAM,
            # Add other methods as needed
        }
        
        if method not in cam_methods:
            raise ValueError(f"Unsupported CAM method: {method}. Supported: {list(cam_methods.keys())}")
        
        self.cam_class = cam_methods[method]
        logger.info(f"Initialized GradCAMExplainer with method: {method}")

    def explain(self, model: Any, image: Union[np.ndarray, torch.Tensor], 
                target_class: int, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for a single image.

        Args:
            model (Any): The PyTorch model.
            image (Union[np.ndarray, torch.Tensor]): The input image.
            target_class (int): The index of the class to explain.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: The raw heatmap as a NumPy array.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a PyTorch tensor.")
            
        try:
            cam = self.cam_class(model=model, target_layers=self.target_layers, use_cuda=torch.cuda.is_available())
            targets = [ClassifierOutputTarget(target_class)]
            
            # The input tensor must have a batch dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            grayscale_cam = cam(input_tensor=image, targets=targets)
            return grayscale_cam[0, :]
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM explanation: {e}")
            raise

    def visualize(self, explanation: np.ndarray, image: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Visualizes the Grad-CAM heatmap on top of the original image.

        Args:
            explanation (np.ndarray): The raw heatmap from the `explain` method.
            image (np.ndarray): The original image (H x W x C, 0-255 RGB).
            **kwargs: Additional parameters, such as alpha for overlay opacity.

        Returns:
            np.ndarray: The visualized image as a NumPy array.
        """
        try:
            # Normalize image to a float range [0, 1] for visualization
            rgb_img = np.float32(image) / 255
            # Use utility function to create the overlay
            visualization = show_cam_on_image(rgb_img, explanation, use_rgb=True)
            return visualization
        except Exception as e:
            logger.error(f"Failed to visualize Grad-CAM heatmap: {e}")
            raise