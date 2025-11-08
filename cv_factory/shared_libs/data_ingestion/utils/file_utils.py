# cv_factory/shared_libs/data_ingestion/utils/file_utils.py (Hardened)

import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_file_paths(directory: str, supported_exts: List[str]) -> List[str]:
    """
    Recursively gets all file paths from a directory that match supported extensions.

    Args:
        directory: The path to the root directory.
        supported_exts: A list of supported file extensions (e.g., [".jpg", ".png"]).

    Returns:
        A list of absolute file paths.
    """
    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return []

    file_paths = []
    # Hardening: Convert extensions to lowercase for robust comparison
    lower_exts = [ext.lower() for ext in supported_exts]
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in lower_exts:
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        logger.warning(f"No files with supported extensions found in {directory}.")
        
    return file_paths

def is_valid_file(file_path: str) -> bool:
    """
    Checks if a file path is a valid and readable file.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file is valid and readable, False otherwise.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.error(f"File does not exist or is not a file: {file_path}")
        return False
    
    if not os.access(file_path, os.R_OK):
        logger.error(f"File is not readable: {file_path}")
        return False
        
    return True