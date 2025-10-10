import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

def clip_gradients(model: nn.Module, max_norm: float) -> None:
    """
    Clips the gradients of a model's parameters to prevent exploding gradients.

    Args:
        model (nn.Module): The PyTorch model.
        max_norm (float): The maximum norm for the gradients.
    """
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        logger.debug("Gradients clipped successfully.")
    except Exception as e:
        logger.error(f"Failed to clip gradients: {e}")
        raise