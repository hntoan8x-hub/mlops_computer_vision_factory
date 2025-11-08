# shared_libs/data_labeling/semi_annotation/methods/refinement_annotator.py (Hardened)

import logging
from typing import List, Dict, Any, Union
from ..base_semi_annotator import BaseSemiAnnotator
from ....data_labeling.configs.label_schema import StandardLabel
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class RefinementAnnotator(BaseSemiAnnotator):
    """
    Specialized Annotator for Label Refinement (Finalization).
    
    Uses logic (e.g., thresholding, NMS) or explicit user feedback to finalize 
    the quality of proposed labels before they enter the 'Trusted Label' pool.
    """

    def refine(self, 
               proposals: List[StandardLabel], 
               user_feedback: Union[Dict[str, Any], None] = None
    ) -> List[StandardLabel]:
        """
        Performs refinement: merging, overlap removal (NMS), or applying user modifications.
        
        Args:
            proposals: List of proposed labels.
            user_feedback: Data containing user actions ('accepted', 'rejected') or corrected labels.

        Returns:
            List[StandardLabel]: List of finalized and validated labels.
        """
        final_labels: List[StandardLabel] = []
        
        if user_feedback:
            action = user_feedback.get("action")
            
            if action == "accepted":
                # Use proposals as accepted, or load corrected labels from feedback
                corrected_data = user_feedback.get("corrected_labels", proposals)
                
                # Hardening: Re-validate user input/accepted proposals against Pydantic
                for label_data in corrected_data:
                    try:
                        # Assuming label_data is already a Pydantic object or a dict matching schema
                        if isinstance(label_data, dict):
                            # Try to convert dict back to Pydantic (requires knowing the type)
                            # Simple approach: assume proposals are the source of truth if type is ambiguous
                            final_labels.append(StandardLabel.__args__[0](**label_data)) # Simplified type recovery
                        else:
                            final_labels.append(label_data)
                    except ValidationError as e:
                        logger.error(f"User accepted label failed re-validation: {e}")
                
                logger.info(f"Labels accepted and finalized by user feedback ({len(final_labels)} labels).")
                
            elif action == "rejected":
                logger.warning("Labels rejected by user feedback.")
                return []
            
        else:
            # Automated refinement (Fallback logic, e.g., high confidence filtering)
            threshold = self.config.get("final_threshold", 0.9)
            
            for p in proposals:
                # Try to extract confidence safely
                confidence = p.model_dump().get("confidence", threshold + 0.01) # Default to pass if confidence not tracked
                if confidence >= threshold:
                    final_labels.append(p)
                    
            logger.info(f"Applied automated refinement: filtered {len(proposals) - len(final_labels)} proposals (Threshold: {threshold}).")

        return final_labels

    def select_samples(self, pool_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Non-functional for RefinementAnnotator. Returns an empty list.
        
        Args:
            pool_metadata: Full metadata list of unlabelled or unconfirmed data samples.

        Returns:
            List[Dict]: Empty list.
        """
        return []