import logging
import json
from typing import Any, Dict

def setup_structured_logging(logger_name: str = 'pipeline_logger'):
    """
    Sets up a structured logging handler that outputs JSON format.

    Args:
        logger_name (str): The name of the logger to configure.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Check if a handler is already configured to avoid duplicates
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(name)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def log_structured_event(logger: logging.Logger, message: str, **kwargs: Dict[str, Any]):
    """
    Logs a structured event with additional key-value pairs.

    Args:
        logger (logging.Logger): The logger instance.
        message (str): The primary log message.
        **kwargs: Additional key-value pairs for the structured log.
    """
    log_dict = {"message": message, **kwargs}
    logger.info(json.dumps(log_dict))