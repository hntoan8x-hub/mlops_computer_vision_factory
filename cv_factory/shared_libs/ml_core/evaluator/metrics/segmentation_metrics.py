# cv_factory/shared_libs/ml_core/evaluator/metrics/segmentation_metrics.py

import numpy as np
import logging
from typing import Dict, Any, Union, List, Tuple, Optional

# Import Base Abstraction
from ..base.base_metric import BaseMetric, MetricValue, InputData

logger = logging.getLogger(__name__)

class DiceCoefficientMetric(BaseMetric):
    """
    A stateful metric class for calculating the Sørensen–Dice Coefficient (Dice Score), 
    often used as the F1 score for image segmentation tasks.

    Accumulates True Positives (TP), False Positives (FP), and False Negatives (FN) per class.
    """
    
    def __init__(self, name: str = 'Dice_Coefficient', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Dice Coefficient Metric.

        Config requires 'num_classes' and 'ignore_index' (optional, typically background).
        """
        super().__init__(name, config)
        self.num_classes = self.config.get('num_classes', 2)
        self.ignore_index = self.config.get('ignore_index', None)
        self.reset()

    def reset(self) -> None:
        """
        Resets the accumulated True Positives (intersection) and total counts 
        (union components) per class.
        """
        self._internal_state = {
            'intersection': np.zeros(self.num_classes, dtype=float),
            'union': np.zeros(self.num_classes, dtype=float)  # sum(A) + sum(B)
        }
        self._is_initialized = True
        logger.debug(f"Metric '{self.name}' state reset.")

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state based on a new batch of segmentation masks.

        Args:
            predictions (np.ndarray): Predicted mask (class indices or one-hot encoded).
            targets (np.ndarray): Ground truth mask (class indices or one-hot encoded).
        """
        if not self._is_initialized:
            self.reset()
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Ensure predictions are class indices if they are logits (by argmax)
        if predictions.ndim > targets.ndim and predictions.shape[-1] == self.num_classes:
             predicted_indices = np.argmax(predictions, axis=-1)
        elif predictions.ndim == targets.ndim:
             predicted_indices = predictions
        else:
             raise ValueError("Unsupported prediction format for Dice Metric update.")
        
        target_indices = targets
        
        # Flatten masks for easier per-class accumulation
        predicted_flat = predicted_indices.flatten()
        target_flat = target_indices.flatten()

        for c in range(self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
                
            # Masks for the current class
            pred_mask = (predicted_flat == c)
            target_mask = (target_flat == c)
            
            # Intersection (True Positives): A AND B
            intersection = np.sum(pred_mask & target_mask)
            
            # Union (Denominator component: A + B): Total predicted + Total target
            # Note: Dice denominator is (sum(A) + sum(B)), not IOU's (A + B - Intersection)
            union = np.sum(pred_mask) + np.sum(target_mask)
            
            self._internal_state['intersection'][c] += intersection
            self._internal_state['union'][c] += union

    def compute(self) -> MetricValue:
        """
        Calculates the final Dice Coefficient score (mean across all non-ignored classes).
        """
        intersection = self._internal_state['intersection']
        union = self._internal_state['union']
        
        epsilon = 1e-6 # Smoothing factor to avoid division by zero
        
        # Dice Score per class: (2 * Intersection) / (Union)
        dice_scores_per_class = np.divide(2. * intersection, union + epsilon, 
                                          out=np.zeros_like(intersection, dtype=float), 
                                          where=(union + epsilon) != 0)
        
        valid_classes = [c for c in range(self.num_classes) if c != self.ignore_index]
        if not valid_classes:
            return 0.0

        mean_dice = np.mean(dice_scores_per_class[valid_classes])
        
        results = {
            self.name: float(mean_dice),
            "Dice_Per_Class": {c: float(dice_scores_per_class[c]) for c in valid_classes}
        }
        return results


class MeanIoUMetric(BaseMetric):
    """
    A stateful metric class for calculating the Mean Intersection over Union (mIoU) 
    for image segmentation.
    """
    
    def __init__(self, name: str = 'Mean_IoU', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the mIoU Metric.
        """
        super().__init__(name, config)
        self.num_classes = self.config.get('num_classes', 2)
        self.ignore_index = self.config.get('ignore_index', None)
        self.reset()

    def reset(self) -> None:
        """
        Resets the accumulated Intersection and Union (Denominator) per class.
        """
        self._internal_state = {
            'intersection': np.zeros(self.num_classes, dtype=float),
            'union': np.zeros(self.num_classes, dtype=float) # (A + B - Intersection)
        }
        self._is_initialized = True
        logger.debug(f"Metric '{self.name}' state reset.")

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state based on a new batch. 
        Logic similar to Dice, but calculating the correct Union for IoU.
        """
        if not self._is_initialized:
            self.reset()
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if predictions.ndim > targets.ndim and predictions.shape[-1] == self.num_classes:
             predicted_indices = np.argmax(predictions, axis=-1)
        elif predictions.ndim == targets.ndim:
             predicted_indices = predictions
        else:
             raise ValueError("Unsupported prediction format for mIoU Metric update.")
        
        target_indices = targets
        
        predicted_flat = predicted_indices.flatten()
        target_flat = target_indices.flatten()

        for c in range(self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
                
            pred_mask = (predicted_flat == c)
            target_mask = (target_flat == c)
            
            # Intersection: A AND B
            intersection = np.sum(pred_mask & target_mask)
            
            # Union: A OR B = (sum(A) + sum(B)) - Intersection
            union = np.sum(pred_mask | target_mask)
            
            self._internal_state['intersection'][c] += intersection
            self._internal_state['union'][c] += union


    def compute(self) -> MetricValue:
        """
        Calculates the final Mean IoU score.
        """
        intersection = self._internal_state['intersection']
        union = self._internal_state['union']
        
        epsilon = 1e-6 
        
        # IoU per class: Intersection / (Union)
        iou_scores_per_class = np.divide(intersection, union + epsilon, 
                                         out=np.zeros_like(intersection, dtype=float), 
                                         where=(union + epsilon) != 0)
        
        valid_classes = [c for c in range(self.num_classes) if c != self.ignore_index]
        if not valid_classes:
            return 0.0
            
        mean_iou = np.mean(iou_scores_per_class[valid_classes])
        
        results = {
            self.name: float(mean_iou),
            "IoU_Per_Class": {c: float(iou_scores_per_class[c]) for c in valid_classes}
        }
        return results