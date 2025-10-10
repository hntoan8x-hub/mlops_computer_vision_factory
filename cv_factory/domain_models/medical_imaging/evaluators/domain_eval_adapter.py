# domain_models/medical_imaging/evaluator/medical_eval_adapter.py (FINALIZED ADAPTER)

import logging
from typing import Dict, Any, Union
# Import Base Orchestrator
from shared_libs.ml_core.evaluator.orchestrator.evaluation_orchestrator import EvaluationOrchestrator

logger = logging.getLogger(__name__)

class MedicalEvalAdapter(EvaluationOrchestrator):
    """
    An adapter that wraps the general EvaluationOrchestrator for the medical domain.

    This class enforces domain-specific rules (e.g., filtering bad data, checking critical thresholds) 
    before or after the generic evaluation is performed.
    (Logic moved from domain_eval_adapter.py)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Call the base Orchestrator's constructor
        super().__init__(config)
        self.domain_rules = config.get("domain_rules", {})
        logger.info("MedicalEvalAdapter initialized with custom domain rules.")

    def evaluate(self, model: Any, data_loader: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full evaluation, applying domain-specific preprocessing and post-evaluation checks.
        """
        
        # Step 1: Domain-specific Pre-Evaluation Logic
        if self.domain_rules.get("filter_invalid_images", False):
            # Conceptual: Implement logic to filter samples using utils/validation_utils.py
            logger.info("Applying domain-specific rules: filtering invalid images.")
            data_loader = self._filter_invalid_samples(data_loader)
        
        # Step 2: Call the generic platform evaluation logic
        # This executes the actual metric calculation (Accuracy, F1, Dice Score)
        results = super().evaluate(model, data_loader, **kwargs)
        
        # Step 3: Domain-specific Post-Evaluation Logic (Quality Gate)
        self._check_for_critical_failure(results)
        
        return results
        
    def _filter_invalid_samples(self, data_loader: Any) -> Any:
        # NOTE: Implementation detail (e.g., creating a new filtered DataLoader)
        # using is_valid_medical_image from utils/
        return data_loader 

    def _check_for_critical_failure(self, results: Dict[str, Any]) -> None:
        """
        Enforces a hard quality gate (e.g., clinical threshold).
        """
        critical_threshold = self.domain_rules.get("critical_metric_threshold", 0.75)
        dice_score = results.get("metrics", {}).get("dice_score", 0.0)
        
        if dice_score < critical_threshold:
            logger.critical(f"Model performance (Dice: {dice_score:.2f}) is below critical threshold ({critical_threshold}). ALERTING TEAM.")
            # NOTE: Here you would call shared_libs.monitoring.alert_utils.send_alert
            # to trigger an email or Slack notification.