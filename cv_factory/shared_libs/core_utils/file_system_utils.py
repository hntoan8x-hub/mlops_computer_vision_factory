# cv_factory/shared_libs/core_utils/file_system_utils.py

import os
import logging
from urllib.parse import urlparse
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# Define recognized Cloud Storage schemes
CLOUD_SCHEMES = ['s3', 'gs', 'azure', 'hdfs']

def check_and_create_dir(path: str) -> None:
    """Checks if a local path exists and creates it if it does not."""
    if is_local_path(path) and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created local directory: {path}")

def is_cloud_uri(uri: str) -> bool:
    """Checks if the given URI is a cloud storage path."""
    parsed = urlparse(uri)
    return parsed.scheme in CLOUD_SCHEMES

def is_local_path(uri: str) -> bool:
    """Checks if the given URI is a local file path."""
    parsed = urlparse(uri)
    return not parsed.scheme or parsed.scheme == 'file'

def get_bucket_and_key(uri: str) -> Union[tuple[str, str], tuple[None, None]]:
    """Extracts bucket and key from a cloud storage URI."""
    if not is_cloud_uri(uri):
        return None, None
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip('/')