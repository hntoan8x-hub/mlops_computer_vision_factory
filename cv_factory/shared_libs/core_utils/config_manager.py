# cv_factory/shared_libs/core_utils/config_manager.py
import yaml
import json
import logging
from typing import Dict, Any, Union, Type, Optional

# Assuming BaseConfig from Pydantic is used for schema validation
from pydantic import BaseModel 

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    A utility class for loading configuration files (YAML/JSON) and applying 
    Pydantic schema validation.
    
    This centralizes all configuration loading logic across the entire Factory.
    """

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """
        Loads configuration data from a YAML or JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            raise

    @staticmethod
    def validate_and_parse(config_data: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        Validates a configuration dictionary against a Pydantic schema.
        
        Args:
            config_data (Dict[str, Any]): The raw dictionary data.
            schema (Type[BaseModel]): The Pydantic model to validate against.
            
        Returns:
            Dict[str, Any]: The validated data dictionary.
        """
        try:
            # Pydantic validation and parsing
            validated_model = schema(**config_data)
            
            # Return as a dictionary for easy use by downstream modules
            return validated_model.dict() 
        except Exception as e:
            logger.error(f"Pydantic schema validation failed: {e}")
            # We don't raise here, as the Orchestrator's validate_config method will wrap 
            # this in a specific InvalidConfigError.
            raise