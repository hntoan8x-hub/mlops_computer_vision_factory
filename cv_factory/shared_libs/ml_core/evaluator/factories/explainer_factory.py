import logging
from typing import Dict, Any, Type, Optional
import torch.nn as nn

from shared_libs.ml_core.evaluator.base.base_explainer import BaseExplainer
from shared_libs.ml_core.evaluator.explainability.gradcam_explainer import GradCAMExplainer
from shared_libs.ml_core.evaluator.explainability.ig_explainer import IGExplainer
from shared_libs.ml_core.evaluator.explainability.lime_explainer import LIMEExplainer
from shared_libs.ml_core.evaluator.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

class ExplainerFactory:
    """
    A factory class for creating different types of model explainers.

    This factory centralizes the creation logic for explainers, ensuring
    easy and consistent instantiation based on a configuration.
    """
    _EXPLAINER_MAP: Dict[str, Type[BaseExplainer]] = {
        "gradcam": GradCAMExplainer,
        "ig": IGExplainer,
        "lime": LIMEExplainer,
        "shap": SHAPExplainer,
    }

    @classmethod
    def create(cls, explainer_type: str, model: nn.Module, config: Optional[Dict[str, Any]] = None) -> BaseExplainer:
        """
        Creates and returns an explainer instance based on its type.

        Args:
            explainer_type (str): The type of explainer to create (e.g., "gradcam", "ig").
            model (nn.Module): The model to be explained.
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the explainer's constructor.

        Returns:
            BaseExplainer: An instance of the requested explainer.

        Raises:
            ValueError: If the specified explainer_type is not supported.
        """
        config = config or {}
        explainer_cls = cls._EXPLAINER_MAP.get(explainer_type.lower())
        
        if not explainer_cls:
            supported_explainers = ", ".join(cls._EXPLAINER_MAP.keys())
            logger.error(f"Unsupported explainer type: '{explainer_type}'. Supported types are: {supported_explainers}")
            raise ValueError(f"Unsupported explainer type: '{explainer_type}'. Supported types are: {supported_explainers}")
        
        logger.info(f"Creating instance of {explainer_cls.__name__}...")
        return explainer_cls(model=model, **config)