# cv_factory/shared_libs/data_ingestion/utils/api_utils.py (Hardened)

import logging
from typing import Dict, Any, Callable
from functools import wraps

import requests

logger = logging.getLogger(__name__)

def set_api_headers(default_headers: Dict[str, str], additional_headers: Dict[str, str]) -> Dict[str, str]:
    """
    Combines default and additional headers for an API request.
    
    Args:
        default_headers: Default headers for the API client.
        additional_headers: Headers specific to a single request.

    Returns:
        The combined headers.
    """
    headers = default_headers.copy()
    headers.update(additional_headers)
    return headers

def handle_api_errors(func: Callable) -> Callable:
    """
    A decorator to handle common API request exceptions and provide clear logging.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            # Hardening: Log status code and truncated response content
            status = e.response.status_code if e.response is not None else 'N/A'
            text = e.response.text[:100] if e.response is not None and e.response.text else 'N/A'
            logger.error(f"HTTP Error: {status}. Response preview: {text}...")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Error: A network error occurred while connecting to the API. {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout Error: The request to the API timed out. {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"An unexpected error occurred during the API request: {e}")
            raise
    return wrapper