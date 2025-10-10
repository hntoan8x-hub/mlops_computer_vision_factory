# domain_models/medical_imaging/evaluator/medical_explain_adapter.py (FINALIZED ADAPTER)

import logging
import numpy as np
from typing import Dict, Any
# Import Base Explainer from shared_libs (assuming GradCAMExplainer is the base)
from shared_libs.ml_core.evaluator.explainability.gradcam_explainer import GradCAMExplainer
# Import Domain-specific visualization utility
from ..utils.visualization_utils import visualize_medical_heatmap

logger = logging.getLogger(__name__)

class MedicalExplainAdapter(GradCAMExplainer):
    """
    An adapter for GradCAMExplainer to handle medical imaging-specific visualizations.

    Overrides the base visualization method to use domain-specific utilities, 
    ensuring heatmaps (like GradCAM) are presented correctly for clinical review.
    (Logic moved from domain_explainability_adapter.py)
    """
    def visualize(self, explanation: np.ndarray, image: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Overrides the base visualize method to use a domain-specific visualization utility.
        
        Args:
            explanation (np.ndarray): The raw heatmap (e.g., GradCAM output).
            image (np.ndarray): The original medical image.
        
        Returns:
            np.ndarray: The visualized image with overlaid heatmap.
        """
        logger.info("Applying medical imaging-specific visualization for heatmap overlay.")
        
        # Delegation to the utility function in the same domain
        return visualize_medical_heatmap(image, explanation, **kwargs)