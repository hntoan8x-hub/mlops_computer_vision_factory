import yaml
import logging
from typing import Type, Any, Dict
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

def load_and_validate_config(config_path: str, schema: Type[BaseModel]) -> BaseModel:
    """
    Loads a YAML configuration file and validates it against a Pydantic schema.

    Args:
        config_path (str): The path to the YAML configuration file.
        schema (Type[BaseModel]): The Pydantic BaseModel class for validation.

    Returns:
        BaseModel: A validated Pydantic object representing the configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
        ValidationError: If the configuration does not match the schema.
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        validated_config = schema.parse_obj(config_dict)
        logger.info(f"Configuration loaded and validated successfully from {config_path}.")
        return validated_config
    
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {config_path}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {config_path}. Error: {e}")
        raise e
    except ValidationError as e:
        logger.error(f"Configuration validation failed for {config_path}. Error: {e}")
        raise e