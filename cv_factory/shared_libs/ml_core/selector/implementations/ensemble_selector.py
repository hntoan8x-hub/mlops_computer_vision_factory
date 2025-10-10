import logging
import numpy as np
from typing import Dict, Any, List, Optional
import torch

from shared_libs.ml_core.selector.base.base_selector import BaseSelector
from shared_libs.ml_core.selector.utils.selection_exceptions import NoValidModelFound

logger = logging.getLogger(__name__)

class EnsembleSelector(BaseSelector):
    """
    Selects a final prediction by ensembling multiple candidate models.

    This class supports various ensembling methods like voting, stacking, or blending.
    Note: This selector's `select` method returns a prediction, not a single model.
    """
    def __init__(self, method: str = 'voting'):
        """
        Initializes the EnsembleSelector.

        Args:
            method (str): The ensembling method ('voting', 'stacking', 'blending').
        """
        if method not in ['voting', 'stacking', 'blending']:
            raise ValueError("Method must be 'voting', 'stacking', or 'blending'.")
        self.method = method
        self.base_models = []
        logger.info(f"Initialized EnsembleSelector with method '{self.method}'.")

    def select(self, candidates: List[Dict[str, Any]], data: Optional[torch.Tensor] = None, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Performs ensembling on the given candidate models to make a final prediction.
        
        Args:
            candidates (List[Dict[str, Any]]): A list of models to be ensembled.
            data (Optional[torch.Tensor]): The input data to make predictions on.

        Returns:
            np.ndarray: The final ensembled prediction.
        """
        if not candidates or len(candidates) < 2:
            raise NoValidModelFound("Ensemble selector requires at least two candidate models.")
        
        if data is None:
            raise ValueError("Ensemble selector requires 'data' to make predictions.")

        all_predictions = []
        for candidate in candidates:
            model = candidate.get('model')
            if model is None:
                logger.warning(f"Candidate '{candidate.get('name')}' has no model instance. Skipping.")
                continue
            
            model.eval()
            with torch.no_grad():
                predictions = torch.nn.functional.softmax(model(data), dim=1)
                all_predictions.append(predictions.cpu().numpy())
        
        if self.method == 'voting':
            # Simple majority voting
            predictions_array = np.array(all_predictions)
            final_predictions = np.argmax(np.sum(predictions_array, axis=0), axis=1)
            return final_predictions
            
        # Add logic for stacking and blending here
        
        return np.array([])
    
    def log_selection(self, selected_model: Optional[Dict[str, Any]], **kwargs: Dict[str, Any]) -> None:
        from shared_libs.ml_core.selector.utils.selection_logging import log_selection_event
        log_selection_event(None, "ensemble_selection", method=self.method)