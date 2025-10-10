# cv_factory/shared_libs/ml_core/trainer/utils/distributed_utils.py

import os
import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

# --- Environment Setup and Initialization ---

def get_world_size() -> int:
    """
    Retrieves the total number of processes participating in the distributed group.

    Returns:
        int: The world size. Returns 1 if distributed environment is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank() -> int:
    """
    Retrieves the rank of the current process within the distributed group.

    Returns:
        int: The global rank. Returns 0 if distributed environment is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def is_main_process() -> bool:
    """
    Checks if the current process is the main (rank 0) process.

    Returns:
        bool: True if the rank is 0, False otherwise.
    """
    return get_rank() == 0

def setup_distributed_environment(backend: str = 'nccl', init_method: Optional[str] = None):
    """
    Initializes the distributed training environment. 
    Reads standard environment variables (e.g., RANK, WORLD_SIZE, MASTER_ADDR) 
    set by platforms like PyTorch Launch, SLURM, or Kubernetes.

    Args:
        backend (str): The communication backend ('nccl' for GPUs, 'gloo' for CPUs).
        init_method (Optional[str]): Method to initialize the process group (e.g., 'env://').
        
    Raises:
        RuntimeError: If distributed environment initialization fails.
    """
    if dist.is_available() and dist.is_initialized():
        logger.info("Distributed environment already initialized.")
        return

    try:
        # Use environment variables if not specified
        if init_method is None:
            init_method = os.environ.get("DISTRIBUTED_INIT_METHOD", "env://")
            
        # Initialize Process Group
        dist.init_process_group(
            backend=backend,
            init_method=init_method
            # PyTorch DDP automatically uses environment variables (RANK, WORLD_SIZE, etc.)
        )
        
        # Set the local device ID for PyTorch
        if torch.cuda.is_available() and backend == 'nccl':
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

        if is_main_process():
            logger.info(f"Distributed environment initialized successfully. Backend: {backend}, World Size: {get_world_size()}")
            
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {e}")
        raise RuntimeError(f"Distributed setup failed: {e}")

# --- Synchronization and Aggregation Utilities ---

def synchronize_between_processes(func_name: str):
    """
    A context manager utility to ensure all processes wait at a barrier.

    Args:
        func_name (str): The name of the function/stage being synchronized (for logging).
    """
    if get_world_size() > 1:
        dist.barrier()
        logger.debug(f"Process {get_rank()} synchronized at: {func_name}")

def reduce_tensor_across_processes(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduces a tensor across all processes (e.g., for aggregating loss or metrics).

    Args:
        tensor (torch.Tensor): The tensor to reduce.
        op (dist.ReduceOp): The operation to apply (e.g., SUM, AVG).
        
    Returns:
        torch.Tensor: The reduced tensor on all ranks (if op is not ReduceOp.SUM).
    """
    if get_world_size() > 1:
        # NOTE: Using all_reduce ensures all processes receive the final aggregated value
        dist.all_reduce(tensor, op=op)
    return tensor

def cleanup_distributed_environment():
    """
    Destroys the distributed process group.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed environment cleaned up.")

# --- Model Wrapping Utility ---

def wrap_model_for_distributed(model: torch.nn.Module) -> torch.nn.Module:
    """
    Wraps the model with DistributedDataParallel (DDP) if the distributed 
    environment is active. Moves the model to the appropriate CUDA device first.

    Args:
        model (torch.nn.Module): The model to wrap.

    Returns:
        torch.nn.Module: The DDP-wrapped model or the original model.
    """
    if get_world_size() > 1 and torch.cuda.is_available() and dist.is_initialized():
        # DDP requires the model to be on the correct device before wrapping
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model.cuda(local_rank)
        
        # NOTE: find_unused_parameters=True can be helpful for models with complex paths (e.g., GANs)
        # but often reduces performance and should be disabled if possible.
        return torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # If not distributed or no CUDA, return the original model
    if torch.cuda.is_available():
        model.cuda() # Move to default GPU if available
    return model