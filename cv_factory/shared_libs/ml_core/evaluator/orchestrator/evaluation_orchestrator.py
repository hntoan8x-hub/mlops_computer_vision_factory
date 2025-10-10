import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union

from shared_libs.ml_core.evaluator.base.base_evaluator import BaseEvaluator
from shared_libs.ml_core.evaluator.factories.metric_factory import MetricFactory
from shared_libs.ml_core.evaluator.factories.explainer_factory import ExplainerFactory
from shared_libs.ml_core.evaluator.utils.visualization_utils import visualize_metrics, visualize_explanations

logger = logging.getLogger(__name__)

class EvaluationOrchestrator(BaseEvaluator):
    """
    Orchestrates the end-to-end model evaluation process.
    
    This class combines metric calculation, explanation generation, and
    visualization based on a unified configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the orchestrator with the evaluation configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the evaluation configuration.
        """
        self.config = config
        self.metrics_to_compute: Dict[str, Callable] = {}
        self.explainers_to_run: List[Any] = []
        self.task_type = config.get("task_type", "classification")
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes the metric functions and explainer instances from the config.
        """
        # Initialize Metrics
        metrics_config = self.config.get("metrics", [])
        for metric_name in metrics_config:
            metric_func = MetricFactory.get_metric(self.task_type, metric_name)
            if metric_func:
                self.metrics_to_compute[metric_name] = metric_func

        # Initialize Explainers
        explainers_config = self.config.get("explainers", [])
        for explainer_config in explainers_config:
            explainer_type = explainer_config.get("type")
            if explainer_type:
                # We need the model instance to initialize the explainer
                # This is a dependency that will be passed in a separate method
                self.explainers_to_run.append(explainer_config)

        logger.info("EvaluationOrchestrator initialized. Components are ready.")

    def evaluate(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                 **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a full evaluation, including metric computation and explanation generation.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            data_loader (torch.utils.data.DataLoader): The test data.
            **kwargs: Additional parameters for evaluation (e.g., visualization flags).

        Returns:
            Dict[str, Any]: A dictionary containing all computed metrics and generated explanations.
        """
        model.eval()
        device = next(model.parameters()).device
        
        # 1. Collect predictions and labels
        y_true_list, y_pred_list, y_pred_proba_list = self._collect_predictions(model, data_loader, device)

        # 2. Compute Metrics
        results: Dict[str, Any] = {"metrics": {}}
        for metric_name, metric_func in self.metrics_to_compute.items():
            if metric_name == "top_k_accuracy":
                k = self.config["metrics_params"].get("top_k", 5)
                score = metric_func(y_true_list, y_pred_proba_list, k=k)
            else:
                score = metric_func(y_true_list, y_pred_list)
            results["metrics"].update(score)
        
        logger.info(f"All metrics computed: {results['metrics']}")

        # 3. Generate Explanations
        explainability_config = self.config.get("explainers", [])
        if explainability_config:
            results["explanations"] = self._run_explainers(model, data_loader, explainability_config)
            logger.info("Explanations generated.")
        
        # 4. Visualization (optional)
        if kwargs.get("visualize_metrics", False):
            visualize_metrics(y_true_list, y_pred_list, self.task_type)
        if kwargs.get("visualize_explanations", False) and "explanations" in results:
            visualize_explanations(results["explanations"])
            
        return results

    def _collect_predictions(self, model, data_loader, device) -> tuple:
        """Helper to run inference and collect predictions, labels, and probabilities."""
        y_true, y_pred, y_pred_proba = [], [], []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                
                outputs = model(images)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0] # Handle models that return a tuple
                    
                probas = torch.nn.functional.softmax(outputs, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(probas, dim=1).cpu().numpy())
                y_pred_proba.extend(probas.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)

    def log_metrics(self, metrics: Dict[str, Any], logger_instance: Any) -> None:
        """Logs metrics to the provided logger instance."""
        if not metrics:
            logger.warning("No metrics to log.")
            return
            
        for key, value in metrics.items():
            logger_instance.log_metric(key, value)
            
        logger.info("Metrics successfully logged.")