import yaml
import logging
from typing import Dict, Any
from pydantic import ValidationError

from shared_libs.ml_core.configs.pipeline_config_schema import PipelineConfig

logger = logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        path (str): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file at {path}: {e}")
        raise

def validate_config(config: Dict[str, Any]) -> PipelineConfig:
    """
    Validates a loaded configuration dictionary against the Pydantic schema.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        PipelineConfig: The validated configuration object.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    try:
        return PipelineConfig.parse_obj(config)
    except ValidationError as e:
        logger.error(f"Invalid configuration schema: {e}")
        raise