import logging
from typing import Dict, Any, List, Optional

from shared_libs.ml_core.selector.base.base_selector import BaseSelector
from shared_libs.ml_core.selector.utils.selection_exceptions import NoValidModelFound

logger = logging.getLogger(__name__)

class MetricBasedSelector(BaseSelector):
    """
    Selects the best model based on a single metric.

    This selector finds the candidate with the highest (or lowest) value for a
    specified metric.
    """
    def __init__(self, metric_name: str, mode: str = 'max'):
        """
        Initializes the selector.

        Args:
            metric_name (str): The name of the metric to use for selection.
            mode (str): The selection mode, either 'max' (for metrics like accuracy)
                        or 'min' (for metrics like loss).
        """
        if mode not in ['max', 'min']:
            raise ValueError("Mode must be 'max' or 'min'.")
        self.metric_name = metric_name
        self.mode = mode
        logger.info(f"Initialized MetricBasedSelector for metric '{self.metric_name}' in '{self.mode}' mode.")

    def select(self, candidates: List[Dict[str, Any]], **kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not candidates:
            raise NoValidModelFound("No candidate models provided.")
            
        best_model = None
        best_score = float('-inf') if self.mode == 'max' else float('inf')
        
        for candidate in candidates:
            metrics = candidate.get('metrics', {})
            score = metrics.get(self.metric_name)
            
            if score is None:
                logger.warning(f"Candidate '{candidate.get('name')}' has no metric '{self.metric_name}'. Skipping.")
                continue

            is_better = (self.mode == 'max' and score > best_score) or \
                        (self.mode == 'min' and score < best_score)
            
            if is_better:
                best_score = score
                best_model = candidate
        
        if best_model is None:
            raise NoValidModelFound(f"No candidate model had the required metric '{self.metric_name}'.")

        logger.info(f"Selected model '{best_model.get('name')}' based on '{self.metric_name}'.")
        return best_model

    def log_selection(self, selected_model: Optional[Dict[str, Any]], **kwargs: Dict[str, Any]) -> None:
        from shared_libs.ml_core.selector.utils.selection_logging import log_selection_event
        log_selection_event(selected_model, "metric_based_selection", self.metric_name, self.mode)