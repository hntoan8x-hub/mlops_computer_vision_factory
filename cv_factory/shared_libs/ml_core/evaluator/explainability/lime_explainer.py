import logging
import numpy as np
from typing import Dict, Any, Union
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch

from shared_libs.ml_core.evaluator.base.base_explainer import BaseExplainer

logger = logging.getLogger(__name__)

class LIMEExplainer(BaseExplainer):
    """
    A concrete explainer for visualizing model decisions using LIME (Local Interpretable Model-agnostic Explanations).
    
    LIME explains the predictions of a complex model by fitting a simple, interpretable model
    locally around the prediction.
    """
    
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Initializes the LIMEExplainer.
        """
        self.explainer = lime_image.LimeImageExplainer()
        logger.info("Initialized LIMEExplainer.")

    def explain(self, model: Any, image: np.ndarray, target_class: int, **kwargs: Dict[str, Any]) -> Any:
        """
        Generates a LIME explanation for a single image's prediction.

        Args:
            model (Any): The machine learning model to be explained.
            image (np.ndarray): The input image (H x W x C) as a NumPy array.
            target_class (int): The index of the target class to explain.
            **kwargs: Additional parameters for the explanation method,
                      e.g., 'num_samples', 'top_labels'.

        Returns:
            Any: The LIME explanation object.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (H x W x C) NumPy array.")
        
        # Define a prediction function for LIME. It must take a batch of images
        # and return the prediction probabilities for each.
        def lime_predict_func(images):
            # LIME returns images in float format, so we need to convert to tensor and normalize
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(model.device)
            # Assuming the model outputs logits or probabilities
            with torch.no_grad():
                logits = model(images_tensor)
                probas = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            return probas

        try:
            explanation = self.explainer.explain_instance(
                image, 
                lime_predict_func, 
                top_labels=kwargs.get("top_labels", 5),
                hide_color=0,
                num_samples=kwargs.get("num_samples", 1000)
            )
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {e}")
            raise

    def visualize(self, explanation: Any, image: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Visualizes the LIME explanation on top of the original image.

        Args:
            explanation (Any): The explanation object from the `explain` method.
            image (np.ndarray): The original image.
            **kwargs: Additional parameters for visualization, e.g., 'num_features'.

        Returns:
            np.ndarray: The visualized image as a NumPy array.
        """
        try:
            # LIME returns the explanation as a mask
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=kwargs.get("num_features", 5),
                hide_rest=False
            )
            
            # Use a utility function to overlay the mask
            visualization = mark_boundaries(temp / 255.0, mask)
            # Re-scale to 0-255 range and convert to uint8
            visualization = (visualization * 255).astype(np.uint8)
            return visualization
        except Exception as e:
            logger.error(f"Failed to visualize LIME explanation: {e}")
            raise