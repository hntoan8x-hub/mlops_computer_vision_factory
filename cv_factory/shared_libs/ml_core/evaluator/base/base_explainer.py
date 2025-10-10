import abc
import numpy as np
import logging
from typing import Dict, Any, Union, List

logger = logging.getLogger(__name__)

class BaseExplainer(abc.ABC):
    """
    Abstract Base Class for all model explainers.

    Defines a standard interface for generating explanations and visualizations
    of a model's predictions.
    """
    
    @abc.abstractmethod
    def explain(self, model: Any, image: Union[np.ndarray, torch.Tensor], 
                target_class: int, **kwargs: Dict[str, Any]) -> Any:
        """
        Generates an explanation for a single image's prediction.

        Args:
            model (Any): The machine learning model to be explained.
            image (Union[np.ndarray, torch.Tensor]): The input image for the explanation.
            target_class (int): The index of the target class to explain.
            **kwargs: Additional parameters for the explanation method.

        Returns:
            Any: The explanation output, which could be a heatmap, a list of
                 important pixels, or a visual overlay.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def visualize(self, explanation: Any, image: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Visualizes the explanation on top of the original image.

        Args:
            explanation (Any): The output from the `explain` method.
            image (np.ndarray): The original image.
            **kwargs: Additional parameters for visualization (e.g., alpha for overlay).

        Returns:
            np.ndarray: The visualized image with the explanation overlay.
        """
        raise NotImplementedError