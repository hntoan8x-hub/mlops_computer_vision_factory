import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Tuple, Union

from sklearn.metrics import roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

def find_optimal_threshold_roc(y_true: Union[np.ndarray, list], y_pred_proba: Union[np.ndarray, list]) -> Tuple[float, float, float]:
    """
    Finds the optimal threshold using the Youden's J statistic from the ROC curve.
    
    Args:
        y_true (Union[np.ndarray, list]): The ground truth labels.
        y_pred_proba (Union[np.ndarray, list]): The predicted probabilities for the positive class.

    Returns:
        Tuple[float, float, float]: A tuple containing (optimal_threshold, sensitivity, specificity).
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        # Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        
        logger.info(f"Optimal threshold found using ROC: {optimal_threshold:.4f}")
        return optimal_threshold, sensitivity, specificity
    except Exception as e:
        logger.error(f"Failed to find optimal threshold using ROC: {e}")
        return 0.5, 0.0, 0.0

def plot_roc_curve(y_true: Union[np.ndarray, list], y_pred_proba: Union[np.ndarray, list]) -> None:
    """
    Plots the ROC curve.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        logger.info("ROC curve plotted successfully.")
    except Exception as e:
        logger.error(f"Failed to plot ROC curve: {e}")