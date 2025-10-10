import logging
import json
from typing import Dict, Any

def get_orchestrator_logger(name: str) -> logging.Logger:
    """Returns a logger instance with a standardized format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "orchestrator": "%(name)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def log_orchestrator_event(logger: logging.Logger, message: str, **kwargs: Dict[str, Any]):
    """Logs a structured event with additional context."""
    log_dict = {"event_type": message, **kwargs}
    logger.info(json.dumps(log_dict))