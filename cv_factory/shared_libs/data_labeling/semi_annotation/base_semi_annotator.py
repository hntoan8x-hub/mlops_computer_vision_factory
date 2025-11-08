# shared_libs/data_labeling/semi_annotation/base_semi_annotator.py (Hardened)

import abc
import logging
from typing import List, Dict, Any, Union
from ....data_labeling.configs.label_schema import StandardLabel

logger = logging.getLogger(__name__)

class BaseSemiAnnotator(abc.ABC):
    """
    Abstract Base Class (ABC) for all Semi-Annotation (Human-in-the-Loop, HITL) methods.
    
    Defines the contract for refining or selecting labels/samples in an iterative process.

    Attributes:
        config (Dict[str, Any]): The configuration dictionary for the specific method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base semi-annotator with configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        # Hardening: Check for required confidence threshold for refinement methods
        if self.__class__.__name__ != 'ActiveLearningSelector' and 'final_threshold' not in config:
             logger.warning(f"{self.__class__.__name__} is missing 'final_threshold' config.")


    @abc.abstractmethod
    def refine(self, 
               proposals: List[StandardLabel], 
               user_feedback: Union[Dict[str, Any], None] = None
    ) -> List[StandardLabel]:
        """
        Performs refinement or finalization of proposed labels based on logic or user feedback.
        
        Args:
            proposals: List of initial proposed labels (from Auto Annotator).
            user_feedback: Data containing user modifications, acceptance, or rejection signals.

        Returns:
            List[StandardLabel]: List of refined/accepted/finalized labels.
        """
        # Hardening: Add a simple check for input type integrity
        if not all(isinstance(p, StandardLabel.__args__) for p in proposals):
             logger.warning("Refine received non-StandardLabel proposals.")
        raise NotImplementedError

    @abc.abstractmethod
    def select_samples(self, pool_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Selects data samples that should be prioritized for the next round of labeling/review (Active Learning).
        
        Args:
            pool_metadata: Full metadata list of unlabelled or unconfirmed data samples.

        Returns:
            List[Dict]: List of metadata for the selected samples prioritized for labeling.
        """
        # Hardening: Check if pool_metadata is provided
        if not isinstance(pool_metadata, list):
             raise TypeError("pool_metadata must be a list of dictionaries.")
        raise NotImplementedError