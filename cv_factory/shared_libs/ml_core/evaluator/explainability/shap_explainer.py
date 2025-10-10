import logging
import numpy as np
import torch
from typing import Dict, Any, Union
import shap

from shared_libs.ml_core.evaluator.base.base_explainer import BaseExplainer

logger = logging.getLogger(__name__)

class SHAPExplainer(BaseExplainer):
    """
    A concrete explainer for visualizing model decisions using SHAP (SHapley Additive exPlanations).
    
    SHAP uses game theory to explain the output of a model by assigning each input
    feature a value representing its contribution to the prediction.
    """
    
    def __init__(self, model: torch.nn.Module, **kwargs: Dict[str, Any]):
        """
        Initializes the SHAPExplainer.
        
        Args:
            model (torch.nn.Module): The PyTorch model to be explained.
        """
        self.model = model
        self.background_data = kwargs.get("background_data", None)
        if self.background_data is None:
            raise ValueError("SHAP Explainer requires 'background_data' for model agnostic explainers.")
            
        # The SHAP explainer needs a function that takes a batch of images and returns the model output
        def model_output_func(images_tensor):
            with torch.no_grad():
                outputs = self.model(images_tensor)
                return outputs.cpu().numpy()

        # Initialize SHAP explainer
        # We use the KernelExplainer for model-agnostic explanations
        self.explainer = shap.KernelExplainer(model_output_func, self.background_data)
        
        logger.info("Initialized SHAP Explainer.")

    def explain(self, model: Any, image: np.ndarray, target_class: int, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Generates SHAP values for a single image's prediction.

        Args:
            model (Any): The model to be explained.
            image (np.ndarray): The input image (H x W x C).
            target_class (int): The index of the class to explain.
            **kwargs: Additional parameters for SHAP explanation.

        Returns:
            np.ndarray: The SHAP values as a NumPy array.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (H x W x C) NumPy array.")

        # SHAP requires the input to be in a specific format for explanation
        # For KernelExplainer, the input should be a single sample
        input_for_shap = image.reshape(1, *image.shape)

        try:
            # Get the SHAP values
            shap_values = self.explainer.shap_values(input_for_shap)
            return shap_values[target_class][0]
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            raise

    def visualize(self, explanation: np.ndarray, image: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Visualizes the SHAP explanation on top of the original image.

        Args:
            explanation (np.ndarray): The SHAP values from the `explain` method.
            image (np.ndarray): The original image (H x W x C).
            **kwargs: Additional visualization parameters.

        Returns:
            np.ndarray: The visualized image as a NumPy array.
        """
        try:
            # We use a SHAP utility function for clean visualization
            # SHAP visualization functions often plot directly. We'll use a trick
            # to get the numpy array from the plot.
            shap.image_plot(explanation, image, show=False)
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data
        except Exception as e:
            logger.error(f"Failed to visualize SHAP explanation: {e}")
            raise