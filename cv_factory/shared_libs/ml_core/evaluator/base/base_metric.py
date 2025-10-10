# cv_factory/shared_libs/ml_core/evaluator/base/base_metric.py

import abc
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Standard type hints for metric results and input data
MetricValue = Union[float, Dict[str, float]]
InputData = Any # Usually model predictions and ground truths

class BaseMetric(abc.ABC):
    """
    Abstract Base Class (ABC) for all stateful evaluation metrics.

    This class enforces a standard interface for accumulating, computing, and
    resetting metric values, suitable for both training loop evaluation and 
    post-deployment performance monitoring.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the base metric.
        
        Args:
            name (str): The display name of the metric (e.g., 'F1_Score', 'mAP').
            config (Optional[Dict[str, Any]]): Configuration for the metric 
                                               (e.g., 'threshold', 'num_classes', 'iou_threshold').
        """
        self.name = name
        self.config = config if config is not None else {}
        self._is_initialized = False
        self._internal_state: Dict[str, Any] = {}
        self.reset()
        
    @abc.abstractmethod
    def update(self, predictions: InputData, targets: InputData, **kwargs) -> None:
        """
        Updates the internal state of the metric based on a new batch of predictions and targets.

        This method should accumulate necessary statistics (True Positives, False Negatives, etc.).

        Args:
            predictions (InputData): Model's raw or processed output.
            targets (InputData): Ground truth labels or annotations.
            **kwargs: Optional batch-specific information (e.g., sample weights).
        """
        if not self._is_initialized:
            logger.warning(f"Metric '{self.name}' is being updated before initialization.")
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self) -> MetricValue:
        """
        Calculates the final metric value based on the accumulated internal state.
        
        Returns:
            MetricValue: The resulting value(s) of the metric.
        """
        if not self._is_initialized:
            raise RuntimeError(f"Cannot compute metric '{self.name}'. Call reset() or update() first.")
        raise NotImplementedError
        
    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the metric to zero or its initial values.
        
        This must be called before starting a new epoch or evaluation run.
        """
        # Example of basic state reset
        self._internal_state = {}
        self._is_initialized = True
        logger.debug(f"Metric '{self.name}' state reset.")
        # Subclasses must implement task-specific state reset (e.g., self.tp = 0)
        raise NotImplementedError

    # --- Utility Methods ---

    def get_config(self) -> Dict[str, Any]:
        """Returns the metric's configuration."""
        return self.config
        
    def get_name(self) -> str:
        """Returns the metric's name."""
        return self.name