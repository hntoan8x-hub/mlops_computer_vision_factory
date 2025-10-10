import logging
from typing import Dict, Any, List, Optional
import datetime

from shared_libs.ml_core.selector.base.base_selector import BaseSelector
from shared_libs.ml_core.selector.utils.selection_exceptions import NoValidModelFound

logger = logging.getLogger(__name__)

class RuleBasedSelector(BaseSelector):
    """
    Selects a model based on a set of predefined rules.

    Rules can include minimum metric thresholds, model creation dates, or specific tags.
    """
    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initializes the selector with a list of rules.

        Args:
            rules (List[Dict[str, Any]]): A list of rule dictionaries.
                                           Example: [{'metric': 'accuracy', 'op': '>=', 'value': 0.8}]
        """
        self.rules = rules
        logger.info("Initialized RuleBasedSelector with custom rules.")

    def select(self, candidates: List[Dict[str, Any]], **kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not candidates:
            raise NoValidModelFound("No candidate models provided.")

        for candidate in candidates:
            if self._satisfies_all_rules(candidate):
                logger.info(f"Selected model '{candidate.get('name')}' as it satisfies all rules.")
                return candidate
        
        raise NoValidModelFound("No candidate model satisfies all the defined rules.")

    def _satisfies_all_rules(self, candidate: Dict[str, Any]) -> bool:
        """
        Checks if a candidate model satisfies all rules.
        """
        for rule in self.rules:
            if not self._satisfies_rule(candidate, rule):
                logger.debug(f"Candidate '{candidate.get('name')}' failed rule: {rule}")
                return False
        return True

    def _satisfies_rule(self, candidate: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        metric_name = rule.get('metric')
        op = rule.get('op')
        value = rule.get('value')
        
        if metric_name:
            candidate_metric = candidate.get('metrics', {}).get(metric_name)
            if candidate_metric is None:
                return False
            if op == '>=': return candidate_metric >= value
            if op == '<=': return candidate_metric <= value
            if op == '==': return candidate_metric == value
        
        # Add more rule types (e.g., 'date', 'tag') here
        
        return False

    def log_selection(self, selected_model: Optional[Dict[str, Any]], **kwargs: Dict[str, Any]) -> None:
        from shared_libs.ml_core.selector.utils.selection_logging import log_selection_event
        log_selection_event(selected_model, "rule_based_selection", rules=self.rules)