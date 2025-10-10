import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def create_metadata_from_ids(ids: List[str]) -> List[Dict[str, Any]]:
    """
    Creates a list of metadata dictionaries from a list of IDs.
    
    Args:
        ids (List[str]): A list of unique IDs (e.g., image filenames, patient IDs).
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each contains a single ID.
    """
    return [{"id": _id} for _id in ids]

def attach_metadata_to_results(results: List[Dict[str, Any]], metadata_store: Dict[Any, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attaches full metadata to a list of search results.
    
    Args:
        results (List[Dict[str, Any]]): The list of search results (e.g., from a FAISS search).
        metadata_store (Dict[Any, Dict[str, Any]]): A dictionary mapping IDs to their full metadata.
        
    Returns:
        List[Dict[str, Any]]: The results list with a 'metadata' key added to each item.
    """
    for item in results:
        item['metadata'] = metadata_store.get(item.get('id'), {})
    return results