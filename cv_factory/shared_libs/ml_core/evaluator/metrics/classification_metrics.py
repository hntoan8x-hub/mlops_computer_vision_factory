# cv_factory/shared_libs/ml_core/evaluator/metrics/classification_metrics.py

import numpy as np
import logging
from typing import Dict, Any, Union, List, Optional

# Import Base Abstraction
from ..base.base_metric import BaseMetric, MetricValue, InputData

logger = logging.getLogger(__name__)

class AccuracyMetric(BaseMetric):
    """
    A stateful metric class for calculating classification accuracy.

    Accumulates total correct predictions and total samples across multiple batches.
    """
    
    def __init__(self, name: str = 'Accuracy', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Accuracy Metric.
        """
        super().__init__(name, config)
        self.reset()

    def reset(self) -> None:
        """
        Resets the accumulated state for a new evaluation run.
        """
        self._internal_state = {
            'total_correct': 0,
            'total_samples': 0
        }
        self._is_initialized = True
        logger.debug(f"Metric '{self.name}' state reset.")

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state with a new batch.

        Args:
            predictions (np.ndarray): Predicted class indices or logits (must be converted to indices).
            targets (np.ndarray): Ground truth class indices.
        """
        if not self._is_initialized:
            self.reset()
            
        # Ensure inputs are NumPy arrays for consistent handling
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Determine predicted indices if predictions are logits/scores
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            predicted_classes = np.argmax(predictions, axis=-1)
        else:
            predicted_classes = predictions
            
        # Basic validation
        if predicted_classes.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predicted_classes.shape} vs targets {targets.shape}")

        # Accumulate state
        correct = np.sum(predicted_classes == targets)
        self._internal_state['total_correct'] += correct
        self._internal_state['total_samples'] += targets.size

    def compute(self) -> float:
        """
        Calculates the final accuracy score.
        """
        if self._internal_state['total_samples'] == 0:
            logger.warning(f"Attempted to compute metric '{self.name}' with zero samples.")
            return 0.0
            
        accuracy = self._internal_state['total_correct'] / self._internal_state['total_samples']
        return float(accuracy)


class F1ScoreMetric(BaseMetric):
    """
    A stateful metric class for calculating the F1 Score, primarily for 
    binary or multi-class scenarios using averaging.
    """
    
    def __init__(self, name: str = 'F1_Score', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the F1 Score Metric.

        Config should typically include 'average' (e.g., 'macro', 'weighted', 'binary') 
        and 'num_classes'.
        """
        super().__init__(name, config)
        self.average = self.config.get('average', 'macro')
        self.num_classes = self.config.get('num_classes', 2)
        self.reset()

    def reset(self) -> None:
        """
        Resets the accumulated True Positives (TP), False Positives (FP), and 
        False Negatives (FN) per class.
        """
        self._internal_state = {
            'tp': np.zeros(self.num_classes, dtype=int),
            'fp': np.zeros(self.num_classes, dtype=int),
            'fn': np.zeros(self.num_classes, dtype=int),
            'supports': np.zeros(self.num_classes, dtype=int)
        }
        self._is_initialized = True
        logger.debug(f"Metric '{self.name}' state reset.")

    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state based on a new batch. 
        Calculates TP, FP, FN for each class.
        """
        if not self._is_initialized:
            self.reset()

        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Convert logits/scores to predicted class indices
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            predicted_classes = np.argmax(predictions, axis=-1)
        else:
            predicted_classes = predictions
        
        # Update TP, FP, FN per class
        for c in range(self.num_classes):
            is_target = targets == c
            is_predicted = predicted_classes == c
            
            self._internal_state['tp'][c] += np.sum(is_target & is_predicted)
            self._internal_state['fp'][c] += np.sum(~is_target & is_predicted)
            self._internal_state['fn'][c] += np.sum(is_target & ~is_predicted)
            self. _internal_state['supports'][c] += np.sum(is_target)

    def compute(self) -> MetricValue:
        """
        Calculates the F1 Score based on the configured averaging method.
        """
        tp = self._internal_state['tp']
        fp = self._internal_state['fp']
        fn = self._internal_state['fn']
        supports = self._internal_state['supports']

        # Calculate Precision and Recall per class
        precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        
        # Calculate F1 per class
        f1_scores = np.divide(2 * precision * recall, (precision + recall), 
                              out=np.zeros_like(tp, dtype=float), 
                              where=(precision + recall) != 0)

        if self.average == 'macro':
            # Simple average of F1 scores across classes
            final_score = np.mean(f1_scores)
        elif self.average == 'weighted':
            # Average weighted by the support (number of true instances) for each class
            total_support = np.sum(supports)
            if total_support == 0:
                final_score = 0.0
            else:
                final_score = np.sum(f1_scores * supports) / total_support
        elif self.average == 'binary' and self.num_classes == 2:
            # For binary classification, typically reports the score for the positive class (index 1)
            final_score = f1_scores[1]
        else:
            raise ValueError(f"Unsupported averaging method or configuration: {self.average}")
            
        return float(final_score)