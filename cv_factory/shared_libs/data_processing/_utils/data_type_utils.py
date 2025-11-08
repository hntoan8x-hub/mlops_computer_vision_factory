# shared_libs/data_processing/_utils/data_type_utils.py

import numpy as np
from typing import Any, List, Union

PreprocessingInput = Union[Any, List[Any]]

def is_video_data(data: PreprocessingInput) -> bool:
    """
    Checks if the input data structure represents a video sequence.

    A video sequence is typically represented as a 4D NumPy array (T x H x W x C) 
    or a list of such arrays (batch).

    Args:
        data (PreprocessingInput): The input data structure (e.g., NumPy array or list).

    Returns:
        bool: True if the input appears to be video data, False otherwise.
    """
    if isinstance(data, np.ndarray):
        # Check if it's a 4D array (T x H x W x C)
        return data.ndim == 4
    elif isinstance(data, list) and data:
        # Check if the first element is a 4D array
        return isinstance(data[0], np.ndarray) and data[0].ndim == 4
    return False