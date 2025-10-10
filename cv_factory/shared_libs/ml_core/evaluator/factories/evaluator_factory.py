import logging
from typing import Dict, Any, Type, Optional

from shared_libs.ml_core.evaluator.base.base_evaluator import BaseEvaluator
from shared_libs.ml_core.evaluator.metrics.classification_metrics import (
    compute_classification_metrics,
    compute_top_k_accuracy,
)
from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator

logger = logging.getLogger(__name__)

class EvaluatorFactory:
    """
    A factory class for creating high-level evaluators based on task type.

    This factory is responsible for instantiating the correct orchestrator
    for a given evaluation task.
    """
    _EVALUATOR_MAP: Dict[str, Type[BaseEvaluator]] = {
        "evaluation_orchestrator": EvaluationOrchestrator,
        # In a real-world scenario, you might have specific evaluators here for
        # different tasks, e.g., ClassificationEvaluator, DetectionEvaluator.
        # But for this architecture, we use a single orchestrator.
    }

    @classmethod
    def create(cls, evaluator_type: str, config: Optional[Dict[str, Any]] = None) -> BaseEvaluator:
        """
        Creates and returns an evaluator instance based on its type.

        Args:
            evaluator_type (str): The type of evaluator to create.
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters.

        Returns:
            BaseEvaluator: An instance of the requested evaluator.

        Raises:
            ValueError: If the specified evaluator_type is not supported.
        """
        config = config or {}
        evaluator_cls = cls._EVALUATOR_MAP.get(evaluator_type.lower())
        
        if not evaluator_cls:
            supported_evaluators = ", ".join(cls._EVALUATOR_MAP.keys())
            logger.error(f"Unsupported evaluator type: '{evaluator_type}'. Supported types are: {supported_evaluators}")
            raise ValueError(f"Unsupported evaluator type: '{evaluator_type}'. Supported types are: {supported_evaluators}")
        
        logger.info(f"Creating instance of {evaluator_cls.__name__}...")
        return evaluator_cls(**config)