import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], labels: List[str]) -> None:
    """
    Plots a confusion matrix using seaborn.

    Args:
        y_true (Union[np.ndarray, list]): The ground truth labels.
        y_pred (Union[np.ndarray, list]): The predicted labels.
        labels (List[str]): The class labels.
    """
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.show()
        logger.info("Confusion matrix plotted successfully.")
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        
def plot_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a heatmap on an image to visualize attention.

    Args:
        image (np.ndarray): The original image (H x W x 3).
        heatmap (np.ndarray): The heatmap (H x W) with values in [0, 1].
        alpha (float): The transparency of the heatmap overlay.

    Returns:
        np.ndarray: The resulting image with the heatmap overlay.
    """
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        image_rgb = np.float32(image) / 255
        
        # Overlay the heatmap on the image
        overlaid_img = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
        overlaid_img = np.uint8(255 * overlaid_img)
        
        logger.info("Heatmap overlay created successfully.")
        return overlaid_img
    except Exception as e:
        logger.error(f"Failed to create heatmap overlay: {e}")
        return image