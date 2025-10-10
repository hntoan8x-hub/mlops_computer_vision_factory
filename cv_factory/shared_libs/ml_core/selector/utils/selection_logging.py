import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def log_selection_event(selected_model: Optional[Dict[str, Any]], selection_type: str, **kwargs: Dict[str, Any]) -> None:
    """
    Logs a structured event about the model selection.

    Args:
        selected_model (Optional[Dict[str, Any]]): The selected model's metadata.
        selection_type (str): The type of selection performed (e.g., "metric_based").
        **kwargs: Additional context to log.
    """
    if selected_model:
        log_message = f"Model '{selected_model.get('name')}' (version: {selected_model.get('version')}) selected."
        log_data = {
            "selection_status": "success",
            "selected_model": selected_model,
            "selection_type": selection_type,
            **kwargs
        }
    else:
        log_message = "No model was selected based on the criteria."
        log_data = {
            "selection_status": "failed",
            "selection_type": selection_type,
            **kwargs
        }
    
    logger.info(log_message, extra=log_data)