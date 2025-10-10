import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Saves a training checkpoint to a specified path.

    Args:
        state (Dict[str, Any]): A dictionary containing the state to be saved,
                                including model_state_dict, optimizer_state_dict, etc.
        path (str): The file path to save the checkpoint.
    """
    try:
        torch.save(state, path)
        logger.info(f"Checkpoint successfully saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {e}")
        raise

def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Loads a training checkpoint from a specified path.

    Args:
        path (str): The file path to the checkpoint.

    Returns:
        Dict[str, Any]: The loaded state dictionary.
    """
    try:
        state = torch.load(path, map_location=lambda storage, loc: storage)
        logger.info(f"Checkpoint successfully loaded from {path}")
        return state
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {path}: {e}")
        raise