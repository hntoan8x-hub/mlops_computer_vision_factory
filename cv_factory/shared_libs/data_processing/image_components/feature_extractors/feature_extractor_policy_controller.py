# shared_libs/data_processing/image_components/feature_extractors/feature_extractor_policy_controller.py

import logging
from typing import List, Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class FeatureExtractorPolicyController:
    """
    Policy Controller for selecting feature extraction steps dynamically.

    Supports conditional execution and governance rules (e.g., ensuring only one 
    deep learning embedder runs) based on external metadata.
    """

    def __init__(self, policy_type: str = "default"):
        """
        Initializes the Policy Controller.

        Args:
            policy_type (str): The policy mode ('default' or 'conditional_metadata').
        
        Raises:
            ValueError: If the policy type is unsupported.
        """
        supported_policies = ["default", "conditional_metadata"]
        policy_type_lower = policy_type.lower()
        
        if policy_type_lower not in supported_policies:
            raise ValueError(f"Unsupported policy type: {policy_type}. Must be one of {supported_policies}.")

        self.policy_type = policy_type_lower
        logger.info(f"Initialized FeatureExtractorPolicyController with policy: {self.policy_type}")

    def select_pipeline(self, pipeline: List[Any], metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Selects the final pipeline steps based on the policy and input metadata.

        Args:
            pipeline (List[Any]): The full list of initialized feature components (Extractors/Embedders).
            metadata (Optional[Dict[str, Any]]): Metadata of the input (e.g., 'source', 'data_type').

        Returns:
            List[Any]: The final list of configured components to run.
        """
        if self.policy_type == "default":
            # Mode 1: Run all enabled components sequentially.
            return pipeline
            
        elif self.policy_type == "conditional_metadata" and metadata:
            # Mode 2: Conditional execution based on input metadata (Advanced Governance)
            
            final_pipeline = []
            
            # Example Metadata Check: If the data source indicates low quality, prioritize simple features.
            is_low_quality = metadata.get('source_quality') == 'low'
            
            # Example Rule: If image is high-resolution, skip SIFT/ORB (computationally expensive)
            is_high_res = metadata.get('resolution', 0) > 4096 

            for component in pipeline:
                component_name = component.__class__.__name__
                
                # Rule 1: Skip expensive keypoint detectors if resolution is too high
                if component_name in ["SIFTExtractor", "ORBExtractor"] and is_high_res:
                    logger.debug(f"Skipping {component_name}: Resolution is too high for efficient keypoint detection.")
                    continue

                # Rule 2: If low quality data, ensure we don't rely solely on deep features (prioritize HOG)
                if is_low_quality and component_name in ["CNNEmbedder", "VITEmbedder"]:
                    # In a real system, you might replace the DL embedder with a simpler one,
                    # or ensure HOG runs first. Here, we just add the component.
                    logger.debug(f"Warning: Low quality data detected. Using {component_name} with caution.")

                # Add component if no skip condition is met
                final_pipeline.append(component)

            return final_pipeline

        # Fallback to default sequential execution
        return pipeline