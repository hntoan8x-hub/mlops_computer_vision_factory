# shared_libs/data_labeling/semi_annotation/methods/active_learning_selector.py (Hardened)

import logging
import random
from typing import List, Dict, Any, Union
import numpy as np
from ..base_semi_annotator import BaseSemiAnnotator
from ....data_labeling.configs.label_schema import StandardLabel

logger = logging.getLogger(__name__)

class ActiveLearningSelector(BaseSemiAnnotator):
    """
    Specialized Annotator for Active Learning: Selects samples for the next round 
    of labeling/review based on criteria (e.g., uncertainty, diversity).
    
    This helps optimize labeling costs by prioritizing samples with the highest information gain.
    """

    def refine(self, proposals: List[StandardLabel], user_feedback: Union[Dict[str, Any], None] = None) -> List[StandardLabel]:
        """
        Non-functional for ActiveLearningSelector. Returns an empty list.
        """
        return []

    def select_samples(self, pool_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Selects data samples to be prioritized for the next round of labeling/review.

        Args:
            pool_metadata: Full metadata list of unlabelled or unconfirmed samples.

        Returns:
            List[Dict]: List of metadata for the selected samples.

        Raises:
            ValueError: If the configured Active Learning method is unsupported.
        """
        # Hardening: Validate required configuration
        selection_size = self.config.get("selection_size", 100)
        selection_method = self.config.get("method", "random")

        if not pool_metadata:
            logger.info("Pool metadata is empty. Returning empty selection.")
            return []

        # Hardening: Ensure selection_size is valid
        if not isinstance(selection_size, int) or selection_size <= 0:
             logger.warning(f"Invalid selection_size: {selection_size}. Defaulting to 100.")
             selection_size = 100
             
        if len(pool_metadata) <= selection_size:
            logger.warning(f"Pool size ({len(pool_metadata)}) is smaller than selection_size ({selection_size}). Selecting all samples.")
            return pool_metadata
        
        
        # --- Logic Lựa Chọn Mẫu ---

        if selection_method == "uncertainty":
            # 1. Uncertainty Sampling
            
            # Hardening: Check if the required score is present
            if 'uncertainty_score' not in pool_metadata[0]:
                 logger.warning("Uncertainty Sampling requested but 'uncertainty_score' is missing in metadata. Defaulting to Random.")
                 return random.sample(pool_metadata, selection_size)

            # Select samples with the highest score (most uncertain)
            sorted_samples = sorted(pool_metadata, key=lambda x: x['uncertainty_score'], reverse=True)
            selected_samples = sorted_samples[:selection_size]
            
            logger.info(f"Active Learning selected {len(selected_samples)} samples using Uncertainty Sampling.")

        elif selection_method == "diversity":
            # 2. Diversity Sampling (Requires embeddings/feature vectors)
            
            # Hardening: Check for required embedding field
            if 'embedding_vector' not in pool_metadata[0]:
                 logger.warning("Diversity Sampling requested but 'embedding_vector' is missing in metadata. Defaulting to Random.")
                 selected_samples = random.sample(pool_metadata, selection_size)
            else:
                 # NOTE: Complex logic (e.g., k-means sampling, core-set) should be implemented here.
                 selected_samples = random.sample(pool_metadata, selection_size)
                 logger.info("Diversity Sampling logic is complex and currently defaulting to Random Selection.")
            
        elif selection_method == "random":
            # 3. Random Sampling (Default/Fallback)
            selected_samples = random.sample(pool_metadata, selection_size)
            logger.info(f"Active Learning selected {len(selected_samples)} samples using Random Sampling.")
            
        else:
            raise ValueError(f"Unsupported Active Learning method: {selection_method}")

        return selected_samples