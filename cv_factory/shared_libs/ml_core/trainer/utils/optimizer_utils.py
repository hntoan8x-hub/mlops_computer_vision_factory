import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

logger = logging.getLogger(__name__)

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Creates an optimizer based on configuration.
    
    Args:
        model (nn.Module): The model whose parameters are to be optimized.
        config (Dict[str, Any]): The optimizer configuration.
        
    Returns:
        torch.optim.Optimizer: The created optimizer.
    """
    optimizer_name = config.get('name', 'Adam').lower()
    lr = config.get('lr', 0.001)
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=config.get('momentum', 0.9))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[Any]:
    """
    Creates a learning rate scheduler based on configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used with the scheduler.
        config (Dict[str, Any]): The scheduler configuration.
        
    Returns:
        Optional[Any]: The created scheduler or None if no scheduler is specified.
    """
    scheduler_name = config.get('name', None)
    if not scheduler_name:
        return None
        
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'steplr':
        return StepLR(optimizer, step_size=config.get('step_size', 10), gamma=config.get('gamma', 0.1))
    elif scheduler_name == 'reducelronplateau':
        return ReduceLROnPlateau(optimizer, mode=config.get('mode', 'min'), factor=config.get('factor', 0.1), patience=config.get('patience', 10))
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")