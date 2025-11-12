import logging
import pickle
from typing import Any

logger = logging.getLogger(__name__)

def save_artifact(obj: Any, path: str) -> None:
    """
    Saves a Python object to a file using pickle.

    Args:
        obj (Any): The object to save.
        path (str): The file path to save to.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Artifact successfully saved to {path}.")
    except Exception as e:
        logger.error(f"Failed to save artifact to {path}: {e}")
        raise

def load_artifact(path: str) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        Any: The loaded object.
    """
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Artifact successfully loaded from {path}.")
        return obj
    except FileNotFoundError:
        logger.error(f"Artifact file not found at {path}.")
        raise
    except Exception as e:
        logger.error(f"Failed to load artifact from {path}: {e}")
        raise