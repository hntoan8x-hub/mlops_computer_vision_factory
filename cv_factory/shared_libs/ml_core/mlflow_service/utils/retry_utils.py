import time
import logging
from typing import Callable
from functools import wraps
from shared_libs.ml_core.mlflow_service.utils.mlflow_exceptions import MLflowAPIError

logger = logging.getLogger(__name__)

def retry(retries: int = 3, delay: int = 5) -> Callable:
    """
    A decorator that retries a function a specified number of times with a delay.

    Args:
        retries (int): The number of times to retry.
        delay (int): The delay in seconds between retries.
    
    Returns:
        Callable: The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {i+1}/{retries} failed for '{func.__name__}': {e}")
                    if i < retries - 1:
                        time.sleep(delay)
                    else:
                        raise MLflowAPIError(f"Function '{func.__name__}' failed after {retries} attempts.")
        return wrapper
    return decorator