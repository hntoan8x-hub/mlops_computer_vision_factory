# cv_factory/shared_libs/ml_core/trainer/utils/checkpoint_utils.py

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Saves a training checkpoint to a specified path using torch.save.

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
    Loads a training checkpoint from a specified path using torch.load.

    Args:
        path (str): The file path to the checkpoint.

    Returns:
        Dict[str, Any]: The loaded state dictionary.
        
    Raises:
        FileNotFoundError: If the checkpoint file is not found.
        Exception: If loading fails for other reasons.
    """
    try:
        # map_location=lambda storage, loc: storage ensures compatibility when loading 
        # a CUDA checkpoint onto a CPU, or vice versa.
        state = torch.load(path, map_location=lambda storage, loc: storage)
        logger.info(f"Checkpoint successfully loaded from {path}")
        return state
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {path}: {e}")
        raise