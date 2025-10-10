import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def visualize_medical_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays a heatmap on a medical image with a specified colormap.

    Args:
        image (np.ndarray): The original medical image.
        heatmap (np.ndarray): The heatmap to be overlaid.
        alpha (float): The transparency of the heatmap.
        colormap (int): The OpenCV colormap to use.

    Returns:
        np.ndarray: The visualized image.
    """
    try:
        # Resize heatmap to match image size
        resized_heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        resized_heatmap = np.uint8(255 * resized_heatmap)

        # Apply colormap and overlay
        colored_heatmap = cv2.applyColorMap(resized_heatmap, colormap)
        overlaid_img = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

        return overlaid_img
    except Exception as e:
        logger.error(f"Failed to visualize medical heatmap: {e}")
        return image

def plot_medical_image(image: np.ndarray, title: str) -> None:
    """
    Plots a medical image with a specific colormap (e.g., 'gray' or 'bone').
    """
    plt.imshow(image, cmap='bone')
    plt.title(title)
    plt.axis('off')
    plt.show()