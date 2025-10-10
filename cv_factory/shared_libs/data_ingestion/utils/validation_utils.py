import logging
import os
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def validate_config_keys(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validates if a dictionary contains all required keys.

    Args:
        config (Dict[str, Any]): The configuration dictionary to validate.
        required_keys (List[str]): A list of keys that must be present.

    Returns:
        bool: True if all required keys are present, False otherwise.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Configuration is missing the following required keys: {missing_keys}")
        return False
    return True