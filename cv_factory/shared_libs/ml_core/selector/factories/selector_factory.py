import logging
from typing import Dict, Any, Type, Optional
import torch.nn as nn

from shared_libs.ml_core.selector.base.base_selector import BaseSelector
from shared_libs.ml_core.selector.implementations.metric_based_selector import MetricBasedSelector
from shared_libs.ml_core.selector.implementations.rule_based_selector import RuleBasedSelector
from shared_libs.ml_core.selector.implementations.ensemble_selector import EnsembleSelector
from shared_libs.ml_core.selector.utils.selection_exceptions import UnsupportedSelectorError

logger = logging.getLogger(__name__)

class SelectorFactory:
    """
    A factory class for creating different types of model selectors.

    This class centralizes the creation logic, allowing for a config-driven
    approach to model selection.
    """
    _SELECTOR_MAP: Dict[str, Type[BaseSelector]] = {
        "metric_based": MetricBasedSelector,
        "rule_based": RuleBasedSelector,
        "ensemble": EnsembleSelector,
    }

    @classmethod
    def create(cls, selector_type: str, config: Optional[Dict[str, Any]] = None) -> BaseSelector:
        """
        Creates and returns a selector instance based on its type.

        Args:
            selector_type (str): The type of selector to create (e.g., "metric_based").
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters
                                                to pass to the selector's constructor.

        Returns:
            BaseSelector: An instance of the requested selector.

        Raises:
            UnsupportedSelectorError: If the specified selector_type is not supported.
        """
        config = config or {}
        selector_cls = cls._SELECTOR_MAP.get(selector_type.lower())
        
        if not selector_cls:
            supported_selectors = ", ".join(cls._SELECTOR_MAP.keys())
            logger.error(f"Unsupported selector type: '{selector_type}'. Supported types are: {supported_selectors}")
            raise UnsupportedSelectorError(f"Unsupported selector type: '{selector_type}'. Supported types are: {supported_selectors}")

        logger.info(f"Creating instance of {selector_cls.__name__}...")
        try:
            return selector_cls(**config)
        except Exception as e:
            logger.error(f"Failed to instantiate selector '{selector_type}' with config {config}: {e}")
            raise