# cv_factory/shared_libs/core_utils/validation_utils.py

import numpy as np
import logging
from typing import Any, List

# Assuming DataIntegrityError is available from core_utils.exceptions
from .exceptions import DataIntegrityError 

logger = logging.getLogger(__name__)

def check_numpy_dimension(arr: np.ndarray, expected_dim: int, name: str = "Input") -> None:
    """
    Verifies that a NumPy array has the expected number of dimensions.
    
    Raises:
        DataIntegrityError: If dimension mismatch occurs.
    """
    if arr.ndim != expected_dim:
        raise DataIntegrityError(
            f"{name} array dimension mismatch. Expected {expected_dim}D, but got {arr.ndim}D."
        )

def check_embedding_size(arr: np.ndarray, expected_size: int) -> None:
    """
    Verifies that the embedding vector size matches the expected size (last dimension).
    """
    if arr.shape[-1] != expected_size:
        raise DataIntegrityError(
            f"Embedding vector size mismatch. Expected size {expected_size}, but got {arr.shape[-1]}."
        )

# Add more checks like check_unique_ids, check_valid_metadata_keys here